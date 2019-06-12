//
//	Copyright (c) 2014 David Wicks, sansumbrella.com
//	All rights reserved.
//
//	Particle Sphere sample application, CPU integration.
//
//	Author: David Wicks
//	License: BSD Simplified
//

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/Rand.h"
#include "cinder/gl/gl.h"
#include "cinder/Log.h"

#include "Queue.hpp"

using namespace ci;
using namespace ci::app;
using namespace std;

/**
 Particle type holds information for rendering and simulation.
 */
struct Particle
{
	vec3 		home;
	vec3 		ppos;
	vec3		pos;
	ColorA		color;
	float		damping;
};

struct Disturbance
{
	vec3 center;
	float force;
};

struct UpdateThreadState
{
	size_t idx;
	size_t count;
	vector<Particle>::iterator first;
	vector<Particle>::iterator last;
	std::chrono::high_resolution_clock::time_point lastUpdateTime;
	SafeQueue<Disturbance> disturbances;
	
	UpdateThreadState():
	idx(0),
	count(0),
	lastUpdateTime(std::chrono::high_resolution_clock::now())
	{}
	
	UpdateThreadState(std::vector<Particle> &particles, size_t idx, size_t count):
		idx(idx),
		lastUpdateTime(std::chrono::high_resolution_clock::now())
	{
		first = particles.begin() + idx;
		
		if (idx + count > particles.size())
		{
			count = particles.size() - idx;
		}
		
		last = particles.begin() + idx + count;
		this->count = count;
	}
	
};

// Home many particles to create. (200k)
const int NUM_PARTICLES = 200e3;

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
	void updateThread(UpdateThreadState &state);
	
private:
	// Particle data on CPU.
	vector<Particle>	_writeParticles;
	vector<Particle>	_readParticles;
	// Buffer holding raw particle data on GPU, written to every update().
	gl::VboRef			_particleVbo;
	// Batch for rendering particles with default shader.
	gl::BatchRef		_particleBatch;
	
	std::mutex _particleBufferLock;
	std::atomic<bool> _running;
	std::vector<shared_ptr<UpdateThreadState>> _threadStates;
	std::vector<std::thread> _threads;
};

void ParticleSphereThreadedCPUApp::setup()
{
	// Create initial particle layout.
	_writeParticles.assign( NUM_PARTICLES, Particle() );
	const float azimuth = 128.0f * M_PI / _writeParticles.size();
	const float inclination = M_PI / _writeParticles.size();
	const float radius = 160.0f;
	vec3 center = vec3( getWindowCenter(), 0.0f );
	for( int i = 0; i < _writeParticles.size(); ++i )
	{	// assign starting values to particles.
		float x = radius * sin( inclination * i ) * cos( azimuth * i );
		float y = radius * cos( inclination * i );
		float z = radius * sin( inclination * i ) * sin( azimuth * i );
		
		auto &p = _writeParticles.at( i );
		p.pos = center + vec3( x, y, z );
		p.home = p.pos;
		p.ppos = p.home + Rand::randVec3() * 10.0f; // random initial velocity
		p.damping = Rand::randFloat( 0.94f, 0.95f );
		p.color = Color( CM_HSV, lmap<float>( i, 0.0f, _writeParticles.size(), 0.0f, 0.66f ), 1.0f, 1.0f );
	}
	
	// Create particle buffer on GPU and copy over data.
	// Mark as streaming, since we will copy new data every frame.
	_readParticles = _writeParticles;
	_particleVbo = gl::Vbo::create( GL_ARRAY_BUFFER, _readParticles, GL_STREAM_DRAW );
	
	// Describe particle semantics for GPU.
	geom::BufferLayout particleLayout;
	particleLayout.append( geom::Attrib::POSITION, 3, sizeof( Particle ), offsetof( Particle, pos ) );
	particleLayout.append( geom::Attrib::COLOR, 4, sizeof( Particle ), offsetof( Particle, color ) );
	
	// Create mesh by pairing our particle layout with our particle Vbo.
	// A VboMesh is an array of layout + vbo pairs
	auto mesh = gl::VboMesh::create( _readParticles.size(), GL_POINTS, { { particleLayout, _particleVbo } } );
#if ! defined( CINDER_GL_ES )
	_particleBatch = gl::Batch::create( mesh, gl::getStockShader( gl::ShaderDef().color() ) );
	gl::pointSize( 1.0f );
#else
	mParticleBatch = gl::Batch::create( mesh, gl::GlslProg::create( loadAsset( "draw_es3.vert" ),
																   loadAsset( "draw_es3.frag" ) ) );
#endif
	
	//
	//	Handle mouse down/move by queueing up "disturbances" to process in update thread
	//
	
	const double Thresh2 = 10 * 10;
	auto canEnqueueDisturbance = [this, Thresh2](const vec3 position) -> bool {
		return true;
//		Disturbance d;
//		if (_disturbances.peek(d)) {
//			return length2(position - d.center) > Thresh2;
//		}
//		return true;
	};
	
	// Disturb particles a lot on mouse down.
	getWindow()->getSignalMouseDown().connect( [this, canEnqueueDisturbance]( MouseEvent event ) {
		vec3 mouse( event.getPos(), 0.0f );
		if (canEnqueueDisturbance(mouse)) {
			const auto d = Disturbance{ mouse, 500.0f };
			for(auto &state : _threadStates)
			{
				state->disturbances.enqueue(d);
			}
		}
	} );
	
	// Disturb particle a little on mouse drag.
	getWindow()->getSignalMouseDrag().connect( [this, canEnqueueDisturbance]( MouseEvent event ) {
		vec3 mouse( event.getPos(), 0.0f );
		if (canEnqueueDisturbance(mouse)) {
			const auto d = Disturbance{mouse, 120.0f};
			for(auto &state : _threadStates)
			{
				state->disturbances.enqueue(d);
			}
		}
	} );
	
	//
	// Start work thread
	//

	_running.store(true);

	const size_t threadCount = max(std::thread::hardware_concurrency(), 1u);
	std::cout << "Creating " << threadCount << " particle threads" << std::endl;

	size_t idx = 0;
	size_t count = static_cast<size_t>(ceil(static_cast<double>(NUM_PARTICLES) / static_cast<double>(threadCount)));
	for (int i = 0; i < threadCount; i++)
	{
		_threadStates.emplace_back(std::make_shared<UpdateThreadState>(_writeParticles, idx, count));
		idx += count;

		_threads.emplace_back(std::thread([this, i](){
			while(_running)
			{
				updateThread(*(_threadStates[i].get()));
			}
		}));
	}
}

#define TARGET_HZ 60.0
#define MAX_WAIT_SECONDS (1.0 / TARGET_HZ)

void ParticleSphereThreadedCPUApp::updateThread(UpdateThreadState &state)
{
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
	double elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - state.lastUpdateTime).count();
	state.lastUpdateTime = now;
	
	if (elapsedSeconds < MAX_WAIT_SECONDS)
	{
		long millis = static_cast<long>(std::floor((MAX_WAIT_SECONDS - elapsedSeconds) * 1000.0));
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	}
	
	while(!state.disturbances.empty())
	{
		Disturbance d = state.disturbances.dequeue();
		
		for( auto i = state.first; i != state.last; ++i) {
			vec3 dir = i->pos - d.center;
			float d2 = length2(dir);
			i->pos += d.force * dir / d2;
		}
	}
	
	// Run Verlet integration on all particles on the CPU.
	double hz = 1.0/elapsedSeconds;
	hz = min(max(hz, 1.0), TARGET_HZ);

	static const int cMax = 90;
	static int c = 0;
	static double hzAcc = 0;
	
	hzAcc += hz;
	c++;
	if (c >= cMax) {
		CI_LOG_D("hz: " << hzAcc / cMax);
		c = 0;
		hzAcc = 0;
	}
	
	float dt2 = static_cast<float>(1.0 / (hz * hz));
	
	for( auto i = state.first; i != state.last; ++i) {
		vec3 vel = (i->pos - i->ppos) * i->damping;
		i->ppos = i->pos;
		vec3 acc = (i->home - i->pos) * 32.0f;
		i->pos += vel + acc * dt2;
	}
	
	{
		std::lock_guard<std::mutex> lock(_particleBufferLock);
		memcpy(_readParticles.data() + state.idx, _writeParticles.data() + state.idx, state.count * sizeof(Particle));
	}
}

void ParticleSphereThreadedCPUApp::update()
{
	// Copy particle data onto the GPU.
	// Map the GPU memory and write over it.
	void *gpuMem = _particleVbo->mapReplace();

	{
		std::lock_guard<std::mutex> lock(_particleBufferLock);
		memcpy( gpuMem, _readParticles.data(), _readParticles.size() * sizeof(Particle) );
	}

	_particleVbo->unmap();
}

void ParticleSphereThreadedCPUApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
	gl::setMatricesWindowPersp( getWindowSize(), 60.0f, 1.0f, 10000.0f );
	gl::enableDepthRead();
	gl::enableDepthWrite();
	
	_particleBatch->draw();
}

void ParticleSphereThreadedCPUApp::cleanup()
{
	_running.store(false);
	for(std::thread &t : _threads)
	{
		t.join();
	}
}

CINDER_APP( ParticleSphereThreadedCPUApp, RendererGl, [] ( App::Settings *settings ) {
	settings->setWindowSize( 1280, 720 );
	settings->setMultiTouchEnabled( false );
} )
