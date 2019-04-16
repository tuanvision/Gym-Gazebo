# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build

# Utility rule file for astra_openni2.

# Include the progress variables for this target.
include ros_astra_camera/CMakeFiles/astra_openni2.dir/progress.make

ros_astra_camera/CMakeFiles/astra_openni2: ros_astra_camera/CMakeFiles/astra_openni2-complete


ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-install
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-mkdir
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-update
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-patch
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-build
ros_astra_camera/CMakeFiles/astra_openni2-complete: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/CMakeFiles
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/CMakeFiles/astra_openni2-complete
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-done

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-install: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing install step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && tar -xjf /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2/Packaging/Final/OpenNI-Linux-2.3.tar.bz2 -C /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/openni2 --strip 1 && mkdir -p /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/openni2/include && ln -fs /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/openni2/Include /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/openni2/include/openni2
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-install

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/openni2
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/tmp
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E make_directory /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-mkdir

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "No download step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E echo_append
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-update: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No update step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E echo_append
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-update

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-patch: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E echo_append
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-patch

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure: ros_astra_camera/astra_openni2/tmp/astra_openni2-cfgcmd.txt
ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-update
ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Performing configure step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && echo "no need to configure"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure

ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-build: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Performing build step for 'astra_openni2'"
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && make release FILTER=On
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera/astra_openni2/OpenNI2 && /usr/bin/cmake -E touch /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-build

astra_openni2: ros_astra_camera/CMakeFiles/astra_openni2
astra_openni2: ros_astra_camera/CMakeFiles/astra_openni2-complete
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-install
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-mkdir
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-download
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-update
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-patch
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-configure
astra_openni2: ros_astra_camera/astra_openni2/src/astra_openni2-stamp/astra_openni2-build
astra_openni2: ros_astra_camera/CMakeFiles/astra_openni2.dir/build.make

.PHONY : astra_openni2

# Rule to build all files generated by this target.
ros_astra_camera/CMakeFiles/astra_openni2.dir/build: astra_openni2

.PHONY : ros_astra_camera/CMakeFiles/astra_openni2.dir/build

ros_astra_camera/CMakeFiles/astra_openni2.dir/clean:
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera && $(CMAKE_COMMAND) -P CMakeFiles/astra_openni2.dir/cmake_clean.cmake
.PHONY : ros_astra_camera/CMakeFiles/astra_openni2.dir/clean

ros_astra_camera/CMakeFiles/astra_openni2.dir/depend:
	cd /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/ros_astra_camera /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera /home/tuanguyen/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/build/ros_astra_camera/CMakeFiles/astra_openni2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ros_astra_camera/CMakeFiles/astra_openni2.dir/depend

