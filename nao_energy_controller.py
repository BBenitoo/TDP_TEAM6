"""my_controller controller."""


from controller import Robot


robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


# Convert time step to seconds for energy calculation (our 'dt')
time_step_seconds = time_step / 1000.0


NUM_MOTORS = 25
motor_names = [
    # Head (2)
    "HeadYaw", "HeadPitch",
    # Left Arm (5)
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LHand",
    # Right Arm (5)
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RHand",
    # Left Leg (6)
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    # Right Leg (6)
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"
]

# 2. ENABLE FEEDBACK
# -----------------------------
# This list will hold the motor device objects
all_motors = [] 
print("Initializing 25 motors for energy feedback...")
for name in motor_names:
    motor = robot.getDevice(name)
    if motor:
        # 1. Enable VELOCITY feedback (to get q_dot)
        motor.enableVelocityFeedback(time_step)
        
        # 2. Enable TORQUE feedback (to get tau)
        motor.enableTorqueFeedback(time_step)
        
        all_motors.append(motor) # Add the motor object to our list
    else:
        print(f"Error: Motor '{name}' not found.")

print("Motor initialization complete.")

# Initialize our total energy variable (our 'E')
total_energy = 0.0



while robot.step(time_step) != -1:
    
    
    instantaneous_power_total = 0.0

    # Iterate through all 25 motor objects
    for motor in all_motors:
        
        # 1. Get Joint Angular Velocity (q_dot)
        # This is your input q̇ᵢ
        q_dot = motor.getVelocity()

        # 2. Get Joint Torque Feedback (tau)
        # This is your input τᵢ
        tau = motor.getTorqueFeedback()

        # 3. Calculate Instantaneous Power for one joint (P_i)
        p_i = tau * q_dot
        
        # 4. Add this joint's power to the robot's total power
        instantaneous_power_total += p_i

    # 5. Calculate Energy consumed in this step (P * dt)
    energy_this_step = instantaneous_power_total * time_step_seconds

    # 6. Add this step's energy to the total ( E = ∫P dt )
    total_energy += energy_this_step

    # Print the results to the console
    print(f"Instantaneous Power: {instantaneous_power_total:.4f} W,  Total Energy: {total_energy:.4f} J")

# Enter here exit cleanup code.
