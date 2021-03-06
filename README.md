# ParticleSphereCPU_threaded
Multi-threaded adaptation of libcinder's ParticleSphereCPU sample. While, *obviously* the correct approach involves moving as much work to GPU as possible, I thought it might be fun to build a multi-threaded CPU implementation. Spawns a thread for each hardware core as reported by `std::thread::hardware_concurrency()` to update a subset of the particle state. WHile the threaded implementation is not *non-blocking* in a strict sense, it does not present any deadlocking writer/reader scenarios.

![Demo Animation](demo.gif)

## BUILD
- Clone this repo into `{CINDER_PATH}/samples/_opengl` and open `xcode/ParticleSphereCPU_threaded.xcodeproj`
- Select the `ParticleSphereCPU_threaded` build scheme and run
