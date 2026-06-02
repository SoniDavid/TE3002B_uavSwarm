FROM osrf/ros:humble-desktop-full

# Add a non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Dev tools + ROS packages needed by tello_vicon
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    libboost-dev \
    libboost-thread-dev \
    python3-transforms3d \
    ros-humble-tf2-tools \
    ros-humble-tf2-ros \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Install bundled Vicon DataStream SDK shared libraries
COPY --chown=root:root ../ros2_ws/src/deps/ros2-vicon-humble/DataStreamSDK_10.1/libViconDataStreamSDK_CPP.so /usr/lib/
COPY --chown=root:root ../ros2_ws/src/deps/ros2-vicon-humble/DataStreamSDK_10.1/libboost_system-mt.so.1.58.0 /usr/lib/
COPY --chown=root:root ../ros2_ws/src/deps/ros2-vicon-humble/DataStreamSDK_10.1/libboost_thread-mt.so.1.58.0 /usr/lib/
COPY --chown=root:root ../ros2_ws/src/deps/ros2-vicon-humble/DataStreamSDK_10.1/libboost_timer-mt.so.1.58.0 /usr/lib/
COPY --chown=root:root ../ros2_ws/src/deps/ros2-vicon-humble/DataStreamSDK_10.1/libboost_chrono-mt.so.1.58.0 /usr/lib/
RUN chmod 0755 \
    /usr/lib/libViconDataStreamSDK_CPP.so \
    /usr/lib/libboost_system-mt.so.1.58.0 \
    /usr/lib/libboost_thread-mt.so.1.58.0 \
    /usr/lib/libboost_timer-mt.so.1.58.0 \
    /usr/lib/libboost_chrono-mt.so.1.58.0 \
    && ldconfig

USER $USERNAME
ENV SHELL /bin/bash
