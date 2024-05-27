# pbstf
This repo implements paper *Position-Based Surface Tension Flow* (SIGGRAPH Aisa 2022). Several cases including:
1. Cubic droplet turns into a sphere without external force
2. Bouncing droplet
3. Faucet dripping

are validated. Here is an animation to be loaded.

<img src="./water_tap.gif"/>

## Download

```bash
git clone git@github.com:f1shel/pbstf.git     # SSH
git clone https://github.com/f1shel/pbstf.git # HTTPS
```

## Dependencies

These dependencies have already been included in the `external` folder.

- [alembic](https://github.com/alembic/alembic)
- [Eigen](https://eigen.tuxfamily.org/)
- [parallel-util](https://github.com/yuki-koyama/parallel-util)
- [timer](https://github.com/yuki-koyama/timer)
- [delaunator-cpp](https://github.com/delfrrr/delaunator-cpp)

## Build

```bash
cd pbstf
mkdir build
cd build
# Step 1: Configure Eigen
mkdir eigen_build
cd eigen_build
cmake ../../external/eigen
cd ..
# Step 2: Configure IMath
mkdir imath_build
cd imath_build
cmake ../../external/IMath -DCMAKE_INSTALL_PREFIX="." -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install --config Release
cd ..
# Step 3: Build the project
cmake .. -DEigen3_DIR="../external/eigen/cmake" -DImath_DIR="./imath_build/lib/cmake/Imath"
cmake --build . --target main --config Release
```

## Run
Intermediate results for visualization will be output in the folder `<executable>/vis`.

## Credit

[https://github.com/yuki-koyama/position-based-fluids](https://github.com/yuki-koyama/position-based-fluids)