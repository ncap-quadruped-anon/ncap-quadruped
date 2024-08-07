# NCAP Quadruped (Deployment)

## 1. Setup communication between local system and robot
- The communication is performed though [LCM](https://lcm-proj.github.io/lcm/).
- Highly encouraged to set up LAN with Ethernet between local PC and robot (A1's Wi-Fi was unreliable in our experience)
- Feel free to use a USB Wi-Fi dongle; The goal is to basically have the local PC be on the robots's IP - 192.168.X.X
- Build instructions can be found [here](https://lcm-proj.github.io/lcm/content/build-instructions.html)
    ```
    # For latest version on Ubuntu/Debian,
    git clone https://github.com/lcm-proj/lcm.git
    cd lcm
    mkdir build && cd build
    cmake ..
    make
    sudo make install
    ```
    
## 2. Build unitree_legged_sdk
- Download from [here](https://github.com/unitreerobotics/unitree_legged_sdk/releases/tag/v3.3.1), unzip and rename to 'unitree_legged_sdk'
    ```
    # after unzipping and changing name to 'unitree_legged_sdk',
    cd unitree_legged_sdk
    mkdir build && cd build
    cmake ..
    make
    ```

## 3. Build Pybind python robot interface
Replace 'user' in ```third_party/interface_sdk/unitree_legged_sdk/CMakeLists.txt``` with your username 
- To set up communication between local system and robot, through a python interface,
    ```
    # export built sdk lib to path
    cd ncap-quadruped/third_party/interface_sdk/unitree_legged_sdk
    mkdir build && cd build
    cmake ..
    make
    sudo make install
    ```

    - Add the generated ```robot_interface.XXX.so``` file to your PYTHONPATH directory (REPO_ROOT)

## 4. Verify the install
- Run these test scripts - ```test_interface.py```, to ensure that the installation went well.
    ```
    # Simple Test
    python -c "
    try:
        import robot_interface
        RobotInterface = robot_interface.RobotInterface
        i = RobotInterface()
        o = i.receive_observaton()
        print('Interface setup okay')
    except Exception as e:
        print(f'Error: {e}, Please add the .so file to your PATH variables to be found or reinstall')
    "
    ```

## 5. Deploy the trained policy
- You are strongly recommended to have a simple suspension system to test out gaits before deploying on ground
- Once the above tests are working, the model can be downloaded and deployed (ensure cfg and chekpoints are present)
    ```
    cd ncap-quadruped/projects/ncap_deploy/scripts
    python deploy_policy.py
    ```

## Licenses

This deployment repo was adapted from [Walk in the Park](https://github.com/ikostrikov/walk_in_the_park).