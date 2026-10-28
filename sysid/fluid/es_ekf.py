import numpy as np

class FishESEKF:
    """
    Error-State Extended Kalman Filter (ES-EKF) for Digital Twin Synchronization.
    Tracks the Nominal State (x) forward using kinematics, and computes an 
    Error State (delta_x) during measurement updates to inject corrections.
    """
    def __init__(self, dt, active_sensors, process_noise_std, sensor_noise_stds, 
                 bias_process_noise=1e-6, initial_covariance=0.1, min_measurement_noise=1e-6):
        self.dt = dt
        
        # Nominal State Vector (11D)
        # 0: Yaw (Global)
        # 1: vX (Local)
        # 2: vY (Local)
        # 3: Omega (Yaw Rate, Local/Global)
        # 4: Alpha (Motor Pos, Local)
        # 5: d_Alpha (Motor Vel, Local)
        # 6: aX Bias (Local)
        # 7: aY Bias (Local)
        # 8: Omega Bias (Local)
        # 9: PosX (Global)
        # 10: PosY (Global)
        self.state_dim = 11

        self.active_sensors = active_sensors
        self.obs_dim = 0
        
        if 'imu' in self.active_sensors: 
            # SOTA Digital Twin Mod: We ONLY measure Omega (Yaw Rate) from the IMU.
            # We explicitly drop aX and aY because integrating them corrupts the multibody fluid dynamics.
            self.obs_dim += 1
        if 'motor_encoder' in self.active_sensors: self.obs_dim += 2
        if 'local_velocity' in self.active_sensors: self.obs_dim += 2
        if 'global_position' in self.active_sensors: self.obs_dim += 2
        if 'heading' in self.active_sensors: self.obs_dim += 1

        # Nominal State (x_nom)
        self.x = np.zeros(self.state_dim)
        
        # Error State Covariance (P)
        self.P = np.eye(self.state_dim) * initial_covariance

        # Error State Process Noise (Q_err)
        self.Q = np.eye(self.state_dim) * process_noise_std**2
        self.Q[6, 6] = bias_process_noise  
        self.Q[7, 7] = bias_process_noise  
        self.Q[8, 8] = bias_process_noise  

        # Construct Measurement Noise (R) and Base Error Observation Matrix (H_err)
        self.R = np.zeros((self.obs_dim, self.obs_dim))
        self.H_err = np.zeros((self.obs_dim, self.state_dim))
        
        idx = 0
        
        if 'imu' in self.active_sensors:
            noise = max(sensor_noise_stds.get('imu', 0.0)**2, min_measurement_noise)
            # Only 1 observation slot for IMU now (Omega)
            self.R[idx, idx] = noise       
            
            # Omega error maps to Omega state (3) and Omega Bias state (8)
            self.H_err[idx, 3] = 1.0
            self.H_err[idx, 8] = 1.0
            
            self.imu_idx = idx
            idx += 1
            
        if 'motor_encoder' in self.active_sensors:
            noise = max(sensor_noise_stds.get('motor_encoder', 0.0)**2, min_measurement_noise)
            self.R[idx, idx] = noise       
            self.R[idx+1, idx+1] = noise   
            self.H_err[idx, 4] = 1.0       
            self.H_err[idx+1, 5] = 1.0     
            self.enc_idx = idx
            idx += 2
            
        if 'local_velocity' in self.active_sensors:
            noise = max(sensor_noise_stds.get('local_velocity', 0.0)**2, min_measurement_noise)
            self.R[idx, idx] = noise       
            self.R[idx+1, idx+1] = noise   
            self.H_err[idx, 1] = 1.0       
            self.H_err[idx+1, 2] = 1.0     
            self.vel_idx = idx
            idx += 2
            
        if 'global_position' in self.active_sensors:
            noise = max(sensor_noise_stds.get('global_position', 0.0)**2, min_measurement_noise)
            self.R[idx, idx] = noise       
            self.R[idx+1, idx+1] = noise   
            self.H_err[idx, 9] = 1.0       
            self.H_err[idx+1, 10] = 1.0    
            self.pos_idx = idx
            idx += 2
            
        if 'heading' in self.active_sensors:
            noise = max(sensor_noise_stds.get('heading', 0.0)**2, min_measurement_noise)
            self.R[idx, idx] = noise
            self.H_err[idx, 0] = 1.0 # Heading maps to Yaw (State 0)
            self.head_idx = idx
            idx += 1

    def predict(self, twin_delta):
        """
        ES-EKF Prediction Step:
        1. Propagate the Nominal State using non-linear kinematics.
        2. Propagate the Error State Covariance using linear Jacobians.
        """
        # --- 1. NOMINAL STATE PROPAGATION ---
        self.x[0] += twin_delta[0]  # Yaw
        self.x[1] += twin_delta[1]  # vX
        self.x[2] += twin_delta[2]  # vY
        self.x[3] += twin_delta[3]  # Omega
        self.x[4] += twin_delta[4]  # Alpha
        self.x[5] += twin_delta[5]  # d_Alpha
        
        c_yaw = np.cos(self.x[0])
        s_yaw = np.sin(self.x[0])
        
        # Local to Global velocity mapping for position integration
        vX_global = self.x[1] * c_yaw - self.x[2] * s_yaw
        vY_global = self.x[1] * s_yaw + self.x[2] * c_yaw
        
        self.x[9] += vX_global * self.dt
        self.x[10] += vY_global * self.dt
        
        # Keep angles bounded in the nominal state
        self.x[0] = (self.x[0] + np.pi) % (2 * np.pi) - np.pi
        self.x[4] = (self.x[4] + np.pi) % (2 * np.pi) - np.pi
        
        # --- 2. ERROR STATE COVARIANCE PROPAGATION ---
        # The Error State (delta_x) mean is perfectly 0, so we only update Covariance.
        # F_err is the Jacobian of the state dynamics evaluated at the nominal state.
        F_err = np.eye(self.state_dim)
        F_err[0, 3] = self.dt   # d_Yaw depends on Omega
        F_err[4, 5] = self.dt   # d_Alpha depends on d_Alpha
        
        # Global position error Jacobians
        F_err[9, 0] = (-self.x[1] * s_yaw - self.x[2] * c_yaw) * self.dt
        F_err[9, 1] = c_yaw * self.dt
        F_err[9, 2] = -s_yaw * self.dt
        
        F_err[10, 0] = (self.x[1] * c_yaw - self.x[2] * s_yaw) * self.dt
        F_err[10, 1] = s_yaw * self.dt
        F_err[10, 2] = c_yaw * self.dt
        
        # SOTA Scaling: Q is process noise per second, so we scale by dt.
        self.P = F_err @ self.P @ F_err.T + (self.Q * self.dt)

    def update(self, z_dict, twin_accel):
        """
        ES-EKF Update Step:
        1. Calculate innovation (residual) from Nominal State.
        2. Compute the optimal Error State (delta_x) via Kalman Gain.
        3. Inject Error State into Nominal State (x = x + delta_x).
        """
        z_real = np.zeros(self.obs_dim)
        z_expected = np.zeros(self.obs_dim)
        
        # --- 1. COMPUTE EXPECTED MEASUREMENTS FROM NOMINAL STATE ---
        if 'imu' in self.active_sensors:
            idx = self.imu_idx
            # z_dict['imu'] is still passed as [aX, aY, omega], but we ONLY use omega (index 2)
            z_real[idx] = z_dict['imu'][2]
            z_expected[idx] = self.x[3] + self.x[8]
            
        if 'motor_encoder' in self.active_sensors:
            idx = self.enc_idx
            z_real[idx:idx+2] = z_dict['motor_encoder']
            z_expected[idx:idx+2] = [self.x[4], self.x[5]]
            
        if 'local_velocity' in self.active_sensors:
            idx = self.vel_idx
            z_real[idx:idx+2] = z_dict['local_velocity']
            z_expected[idx:idx+2] = [self.x[1], self.x[2]]
            
        if 'global_position' in self.active_sensors:
            idx = self.pos_idx
            z_real[idx:idx+2] = z_dict['global_position']
            z_expected[idx:idx+2] = [self.x[9], self.x[10]]

        if 'heading' in self.active_sensors:
            idx = self.head_idx
            z_real[idx] = z_dict['heading']
            z_expected[idx] = self.x[0]

        # --- 2. COMPUTE RESIDUAL ---
        y = z_real - z_expected
        
        # Wrap angular residuals (Only for true angles, NOT angular velocities like Omega)
        if 'motor_encoder' in self.active_sensors:
            y[self.enc_idx] = (y[self.enc_idx] + np.pi) % (2 * np.pi) - np.pi
        if 'heading' in self.active_sensors:
            y[self.head_idx] = (y[self.head_idx] + np.pi) % (2 * np.pi) - np.pi

        # --- 3. FILTER NaN MEASUREMENTS (Surgical Robustness Fix) ---
        is_valid = np.isfinite(y)
        if not np.any(is_valid):
            # If no valid measurements, skip update but return current state
            return self.x, y, np.zeros((self.state_dim, self.obs_dim)), np.diag(self.P)

        y_v = y[is_valid]
        H_v = self.H_err[is_valid, :]
        R_v = self.R[np.ix_(is_valid, is_valid)]

        # --- 4. CALCULATE KALMAN GAIN & ERROR STATE ---
        S = H_v @ self.P @ H_v.T + R_v
        K_v = self.P @ H_v.T @ np.linalg.inv(S)

        # delta_x is the computed Error State
        delta_x = K_v @ y_v
        
        # --- 5. ERROR INJECTION (COMPOSITION) ---
        # We inject the Error State into the Nominal State
        self.x = self.x + delta_x
        
        # Keep angular states bounded upon composition
        self.x[0] = (self.x[0] + np.pi) % (2 * np.pi) - np.pi 
        self.x[4] = (self.x[4] + np.pi) % (2 * np.pi) - np.pi 
        
        # --- 6. COVARIANCE UPDATE ---
        self.P = (np.eye(self.state_dim) - K_v @ H_v) @ self.P

        # Reconstruct full K for logging compatibility
        K = np.zeros((self.state_dim, self.obs_dim))
        K[:, is_valid] = K_v

        return self.x, y, K, np.diag(self.P)


