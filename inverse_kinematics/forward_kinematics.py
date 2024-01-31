import numpy as np


def forward_kinematics(array_of_joint_angles):
    # array_of_joint_angles is the output of get_sin_cos_joint_angles(), basically we can get it by -> arm_observables.joints_pos._raw_callable(mjcf_physics_instance)
    theta1 = np.arctan2(array_of_joint_angles[0][0], array_of_joint_angles[0][1])
    theta2 = np.arctan2(array_of_joint_angles[1][0], array_of_joint_angles[1][1])
    theta3 = np.arctan2(array_of_joint_angles[2][0], array_of_joint_angles[2][1])
    theta4 = np.arctan2(array_of_joint_angles[3][0], array_of_joint_angles[3][1])
    theta5 = np.arctan2(array_of_joint_angles[4][0], array_of_joint_angles[4][1])
    theta6 = np.arctan2(array_of_joint_angles[5][0], array_of_joint_angles[5][1])
    
    mm = 10**-3
    # DH Parameters
    aa = 11*np.pi/72
    sa = np.sin(aa)
    ca = np.cos(aa)
    s2a = np.sin(2*aa)
    c2a = np.cos(2*aa)
    D3 = 0.2073
    D4 = 0.0743
    D5 = 0.0743
    D6 = 0.1687
    e2 = 0.0098
    d4b = D3 + (sa/s2a)*D5
    d5b = (sa/s2a)*D4 + (sa/s2a)*D5
    d6b = (sa/s2a)*D5 + D6
    a =     [0,          0.410,          0,                 0,                 0,                 0                     ] 
    alpha = [np.pi/2,    np.pi,          np.pi/2,           2*aa,              2*aa,              np.pi                 ] 
    d =     [0.2755,     0,              -e2,               -d4b,               -d5b,              -d6b                  ]
    theta = [-theta1,    theta2+np.pi/2, theta3 - np.pi/2,  theta4,            theta5+np.pi,      theta6-(100*np.pi/180)] 

    T = []
    for i in range(len(theta)):
        Ti = np.array([
            [np.cos(theta[i]), -np.sin(theta[i])*np.cos(alpha[i]), np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i])*np.cos(alpha[i]), -np.cos(theta[i])*np.sin(alpha[i]), a[i]*np.sin(theta[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
        T.append(Ti)

    T_total = T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ T[5]

    # print (T_total)

    row , col = T_total.shape
    for i in range(row):
        for j in range(col):
            if abs(T_total[i][j]) < 0.001:
                T_total[i][j] = 0
            
        

    position = T_total[:3, 3]
    orientation = T_total[:3, :3]

    return [position, orientation]

    # print("End effector position:", position)
    # print("End effector orientation:\n", orientation)

def get_angles(array_of_joint_angles):
    thetas = []
    for i in range(6):
        theta = np.arctan2(array_of_joint_angles[i][0], array_of_joint_angles[i][1])
        thetas.append(theta)
    # theta1 = np.arctan2(array_of_joint_angles[0][0], array_of_joint_angles[0][1])
    # theta2 = np.arctan2(array_of_joint_angles[1][0], array_of_joint_angles[1][1])
    # theta3 = np.arctan2(array_of_joint_angles[2][0], array_of_joint_angles[2][1])
    # theta4 = np.arctan2(array_of_joint_angles[3][0], array_of_joint_angles[3][1])
    # theta5 = np.arctan2(array_of_joint_angles[4][0], array_of_joint_angles[4][1])
    # theta6 = np.arctan2(array_of_joint_angles[5][0], array_of_joint_angles[5][1])
    # return np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    return np.stack(thetas)
    # return [theta1, theta2, theta3, theta4, theta5, theta6]