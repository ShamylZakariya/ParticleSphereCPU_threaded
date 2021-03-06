//
//	Copyright (c) 2014 David Wicks, sansumbrella.com
//	All rights reserved.
//
//	Particle Sphere sample application, CPU integration.
//
//	Author: David Wicks
//	License: BSD Simplified
//	Adapted by Shamyl Zakariya for multithreaded experimentation
//

#include "cinder/Log.h"
#include "cinder/Rand.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "Queue.hpp"

using namespace ci;
using namespace ci::app;
using namespace std;

// How many particles to create
const int NUM_PARTICLES = 400e3;

// target simulation rate
const double TARGET_SIMULATION_HZ = 60.0;
const double MAX_SIMULATION_FRAME_DURATION_SECONDS = (1.0 / TARGET_SIMULATION_HZ);

const bool LIVING_DANGEROUSLY = false;
const bool SHUFFLE_PARTICLES = true;

/**
 Particle type holds information for rendering and simulation.
 */
struct Particle {
    vec3 home;
    vec3 ppos;
    vec3 pos;
    ColorA color;
    float damping;
};

struct Disturbance {
    vec3 center;
    float force;
};

struct ThreadState {
    size_t idx;
    size_t count;
    vector<Particle>::iterator first;
    vector<Particle>::iterator last;
    std::chrono::high_resolution_clock::time_point lastUpdateTime;
    std::mutex accessLock;
    SafeQueue<Disturbance> disturbances;
    double updateHz;

    ThreadState()
        : idx(0)
        , count(0)
        , lastUpdateTime(std::chrono::high_resolution_clock::now())
        , updateHz(TARGET_SIMULATION_HZ)
    {
    }

    ThreadState(std::vector<Particle>& particles, size_t idx, size_t count)
        : idx(idx)
        , lastUpdateTime(std::chrono::high_resolution_clock::now())
        , updateHz(TARGET_SIMULATION_HZ)
    {
        first = particles.begin() + idx;

        if (idx + count > particles.size()) {
            count = particles.size() - idx;
        }

        last = particles.begin() + idx + count;
        this->count = count;
    }
};

/**
 Simple particle simulation with Verlet integration and mouse interaction.
 A sphere of particles is deformed by mouse interaction.
 Simulation is run on the CPU in the update() function.
 Designed to have the same behavior as ParticleSphereGPU.
 */

class ParticleSphereThreadedCPUApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void cleanup() override;

private:
    void updateThread(ThreadState& state);

private:
    // particle data which will be written to by worker threads
    vector<Particle> _writeParticles;

    // particle data which will be read from by main thread to pipe to GPU
    vector<Particle> _readParticles;

    // cinder abstractions
    gl::VboRef _particleVbo;
    gl::BatchRef _particleBatch;

    // threading
    std::atomic<bool> _running;
    std::vector<shared_ptr<ThreadState>> _threadStates;
    std::vector<std::thread> _threads;
};

void ParticleSphereThreadedCPUApp::setup()
{
    // Create initial particle layout.
    _writeParticles.assign(NUM_PARTICLES, Particle());
    const float azimuth = 128.0f * M_PI / _writeParticles.size();
    const float inclination = M_PI / _writeParticles.size();
    const float radius = 160.0f;
    vec3 center = vec3(getWindowCenter(), 0.0f);
    for (int i = 0; i < _writeParticles.size(); ++i) {
        // assign starting values to particles.
        float x = radius * sin(inclination * i) * cos(azimuth * i);
        float y = radius * cos(inclination * i);
        float z = radius * sin(inclination * i) * sin(azimuth * i);

        auto& p = _writeParticles.at(i);
        p.pos = center + vec3(x, y, z);
        p.home = p.pos;
        p.ppos = p.home + Rand::randVec3() * 10.0f; // random initial velocity
        p.damping = Rand::randFloat(0.94f, 0.95f);
        p.color = Color(CM_HSV, lmap<float>(i, 0.0f, _writeParticles.size(), 0.0f, 0.66f),
            1.0f, 1.0f);
    }
    
    if (SHUFFLE_PARTICLES)
    {
        std::random_shuffle(_writeParticles.begin(), _writeParticles.end());
    }

    // Create particle buffer on GPU and copy over data.
    // Mark as streaming, since we will copy new data every frame.
    _readParticles = _writeParticles;
    _particleVbo = gl::Vbo::create(GL_ARRAY_BUFFER, _readParticles, GL_STREAM_DRAW);

    // Describe particle semantics for GPU.
    geom::BufferLayout particleLayout;
    particleLayout.append(geom::Attrib::POSITION, 3, sizeof(Particle),
        offsetof(Particle, pos));
    particleLayout.append(geom::Attrib::COLOR, 4, sizeof(Particle),
        offsetof(Particle, color));

    // Create mesh by pairing our particle layout with our particle Vbo.
    // A VboMesh is an array of layout + vbo pairs
    auto mesh = gl::VboMesh::create(static_cast<uint32_t>(_readParticles.size()),
        GL_POINTS, { { particleLayout, _particleVbo } });
#if !defined(CINDER_GL_ES)
    _particleBatch = gl::Batch::create(mesh, gl::getStockShader(gl::ShaderDef().color()));
    gl::pointSize(1.0f);
#else
    mParticleBatch = gl::Batch::create(mesh, gl::GlslProg::create(loadAsset("draw_es3.vert"), loadAsset("draw_es3.frag")));
#endif

    //
    //	Handle mouse down/move by queueing up "disturbances" to process in update thread
    //

    // Disturb particles a lot on mouse down.
    getWindow()->getSignalMouseDown().connect(
        [this](MouseEvent event) {
            vec3 mouse(event.getPos(), 0.0f);
            const auto d = Disturbance { mouse, 500.0f };
            for (auto& state : _threadStates) {
                state->disturbances.enqueue(d);
            }
        });

    // Disturb particle a little on mouse drag.
    getWindow()->getSignalMouseDrag().connect(
        [this](MouseEvent event) {
            vec3 mouse(event.getPos(), 0.0f);
            const auto d = Disturbance { mouse, 120.0f };
            for (auto& state : _threadStates) {
                state->disturbances.enqueue(d);
            }
        });

    //
    // Start work threads to act on slices of the _writeParticle store
    //

    _running.store(true);

    const size_t threadCount = max(std::thread::hardware_concurrency(), 1u);
    std::cout << "Creating " << threadCount << " worker threads" << std::endl;

    size_t idx = 0;
    size_t count = static_cast<size_t>(ceil(static_cast<double>(NUM_PARTICLES) / static_cast<double>(threadCount)));
    for (int i = 0; i < threadCount; i++) {

        auto state = std::make_shared<ThreadState>(_writeParticles, idx, count);
        _threadStates.push_back(state);
        idx += count;

        _threads.emplace_back(std::thread([this, state]() {
            while (_running) {
                updateThread(*(state.get()));
            }
        }));
    }
}

void ParticleSphereThreadedCPUApp::updateThread(ThreadState& state)
{
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - state.lastUpdateTime).count();
    state.lastUpdateTime = now;

    // if we have some lag time, sleep
    if (elapsedSeconds < MAX_SIMULATION_FRAME_DURATION_SECONDS) {
        long millis = static_cast<long>(std::floor(
            (MAX_SIMULATION_FRAME_DURATION_SECONDS - elapsedSeconds) * 1000.0));
        std::this_thread::sleep_for(std::chrono::milliseconds(millis));
    }

    // process queued disturbances to apply
    while (!state.disturbances.empty()) {
        Disturbance d = state.disturbances.dequeue();
        for (auto i = state.first; i != state.last; ++i) {
            vec3 dir = i->pos - d.center;
            float d2 = length2(dir);
            i->pos += d.force * dir / d2;
        }
    }

    // Determine the approximate run hz, then
    // run Verlet integration on all particles on the CPU.
    double hz = 1.0 / elapsedSeconds;
    hz = min(max(hz, 1.0), TARGET_SIMULATION_HZ);
    state.updateHz = (state.updateHz + hz) * 0.5;
    float dt2 = static_cast<float>(1.0 / (state.updateHz * state.updateHz));

    for (auto i = state.first; i != state.last; ++i) {
        vec3 vel = (i->pos - i->ppos) * i->damping;
        i->ppos = i->pos;
        vec3 acc = (i->home - i->pos) * 32.0f;
        i->pos += vel + acc * dt2;
    }

    // lock and write our section of particle state to the appropriate place in the read buffer
    {
        if (LIVING_DANGEROUSLY)
        {
            memcpy(_readParticles.data() + state.idx,
                   _writeParticles.data() + state.idx, state.count * sizeof(Particle));
        }
        else
        {
            std::lock_guard<std::mutex> lock(state.accessLock);
            memcpy(_readParticles.data() + state.idx,
                   _writeParticles.data() + state.idx, state.count * sizeof(Particle));
        }
    }
}

void ParticleSphereThreadedCPUApp::update()
{
    // Copy particle data onto the GPU.
    // Get writeable buffer to GPU memory
    uint8_t* gpuMem = static_cast<uint8_t*>(_particleVbo->mapReplace());

    // for each thread state, if available, copy particle data subsection over
    for (auto& state : _threadStates) {
        if (LIVING_DANGEROUSLY || state->accessLock.try_lock()) {
            memcpy(gpuMem + (state->idx * sizeof(Particle)), _readParticles.data() + state->idx, state->count * sizeof(Particle));
            state->accessLock.unlock();
        }
    }

    _particleVbo->unmap();
}

void ParticleSphereThreadedCPUApp::draw()
{
    gl::clear(Color(0, 0, 0));
    gl::setMatricesWindowPersp(getWindowSize(), 60.0f, 1.0f, 10000.0f);
    gl::enableDepthRead();
    gl::enableDepthWrite();

    _particleBatch->draw();
}

void ParticleSphereThreadedCPUApp::cleanup()
{
    _running.store(false);
    for (std::thread& t : _threads) {
        t.join();
    }
}

CINDER_APP(ParticleSphereThreadedCPUApp, RendererGl,
    [](App::Settings* settings) {
        settings->setHighDensityDisplayEnabled(true);
        settings->setWindowSize(1280, 720);
        settings->setMultiTouchEnabled(false);
    })
