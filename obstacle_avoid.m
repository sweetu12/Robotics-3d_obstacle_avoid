% Robot parameters
l1 = 11;  % Length of link 1
l2 = 10;  % Length of link 2

% Target position in 3D (x, y, z)
p_final = [18, 9, 5];

% Obstacle parameters: center (x, y, z) and radius
obstacle_center = [6, 13, 3];
obstacle_radius = 5.25;

% Initial joint angles: theta1 (base rotation), theta2 (shoulder elevation), theta3 (elbow angle)
theta1 = pi/2;
theta2 = 0;
theta3 = pi;

% Gradient descent parameters
step_size = 0.01;
tolerance = 0.1;
max_iterations = 10000;

% Create a figure for 3D animation
figure;
hold on;
grid on;
axis equal;
xlabel('X-axis'); ylabel('Y-axis'); zlabel('Z-axis');
title('3-Link Spherical Robot Transition with Obstacle Avoidance');
xlim([-25 25]); ylim([-25 25]); zlim([-25 25]);
view(3);

% Plot obstacle as a semi-transparent sphere
[sx, sy, sz] = sphere(20);
surf(obstacle_center(1) + obstacle_radius*sx, obstacle_center(2) + obstacle_radius*sy, obstacle_center(3) + obstacle_radius*sz, 'FaceAlpha',0.3, 'EdgeColor','none', 'FaceColor','r');

% Plot target position
plot3(p_final(1), p_final(2), p_final(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor','g');

% Plot initial robot configuration (base, joint, end-effector)
robot_line = plot3([0,0,0],[0,0,0],[0,0,0], '-o', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor','b');

% Initialize tracer to record the end-effector path (dashed black line)
tracer_line = plot3(NaN, NaN, NaN, 'k--', 'LineWidth', 1.5);
tracer_positions = []; 

% Initialize tracer for the joint position (p1) path (dotted pink line)
joint_tracer_line = plot3(NaN, NaN, NaN, 'm:', 'LineWidth', 1.5);
joint_tracer_positions = []; 



% Gradient descent loop
for iter = 1:max_iterations
    % Compute joint forces via the workspace potential field function
    [F1, F2, F3] = workspace_potential_fields_3d(theta1, theta2, theta3, l1, l2, p_final, obstacle_center, obstacle_radius);
    
    % Update joint angles with a limited step size for stability
    theta1 = theta1 + step_size * sign(F1) * min(abs(F1), 0.1);
    theta2 = theta2 + step_size * sign(F2) * min(abs(F2), 0.1);
    theta3 = theta3 + step_size * sign(F3) * min(abs(F3), 0.1);
    
    % Forward kinematics for visualization
    p1 = [l1*cos(theta1)*cos(theta2), l1*sin(theta1)*cos(theta2), l1*sin(theta2)];
    p2 = [p1(1) + l2*cos(theta1)*cos(theta2+theta3), p1(2) + l2*sin(theta1)*cos(theta2+theta3), p1(3) + l2*sin(theta2+theta3)];
    
    % current positions to tracers
    tracer_positions = [tracer_positions; p2];
    joint_tracer_positions = [joint_tracer_positions; p1];
    
    % Update the robot's line for animation
    set(robot_line, 'XData', [0, p1(1), p2(1)], 'YData', [0, p1(2), p2(2)], 'ZData', [0, p1(3), p2(3)]);
    
    % Update tracer lines with the accumulated positions
    set(tracer_line, 'XData', tracer_positions(:,1), 'YData', tracer_positions(:,2), 'ZData', tracer_positions(:,3));
    set(joint_tracer_line, 'XData', joint_tracer_positions(:,1), 'YData', joint_tracer_positions(:,2), 'ZData', joint_tracer_positions(:,3));
    
    drawnow;
    pause(0.001);
    
    % Check if target is reached
    if norm(p2 - p_final) < tolerance
        disp(['Target reached in ', num2str(iter), ' iterations!']);
        break;
    end
end

if iter == max_iterations
    disp('Max iterations reached without converging to the target.');
end





% Workspace potential fields function

function [F1, F2, F3] = workspace_potential_fields_3d(theta1, theta2, theta3, l1, l2, p_final, obstacle_center, obstacle_radius)
    % Forward kinematics to compute joint positions
    p1 = [l1*cos(theta1)*cos(theta2), l1*sin(theta1)*cos(theta2), l1*sin(theta2)];
    p2 = [p1(1) + l2*cos(theta1)*cos(theta2+theta3), p1(2) + l2*sin(theta1)*cos(theta2+theta3), p1(3) + l2*sin(theta2+theta3)];
      
    % Attractive force (pulling the end-effector toward the target)
    k_att = 1.0;
    F_att = -k_att * (p2 - p_final);
    
    % Repulsive force (pushing away from the obstacle when too close)
    k_rep = 1.0;
    dist_to_obstacle = norm(p2 - obstacle_center);
    if dist_to_obstacle < obstacle_radius
        F_rep = k_rep * ((1/dist_to_obstacle - 1/obstacle_radius) / (dist_to_obstacle^2)) * (p2 - obstacle_center);
    else
        F_rep = [0, 0, 0];
    end
    

    % Total workspace force
    F_total = F_att + F_rep;
    
    % Compute the Jacobian (3x3 matrix)
    % Partial derivative with respect to theta1:
    dp_dtheta1 = [-sin(theta1)*(l1*cos(theta2)+l2*cos(theta2+theta3)), cos(theta1)*(l1*cos(theta2)+l2*cos(theta2+theta3)), 0];
    % Partial derivative with respect to theta2:
    dp_dtheta2 = [-l1*cos(theta1)*sin(theta2)-l2*cos(theta1)*sin(theta2+theta3), -l1*sin(theta1)*sin(theta2)-l2*sin(theta1)*sin(theta2+theta3), l1*cos(theta2)+l2*cos(theta2+theta3)];
    % Partial derivative with respect to theta3:
    dp_dtheta3 = [-l2*cos(theta1)*sin(theta2+theta3), -l2*sin(theta1)*sin(theta2+theta3), l2*cos(theta2+theta3)];
               
    J = [dp_dtheta1; dp_dtheta2; dp_dtheta3]';
    
    % Map the workspace force to joint torques via the transpose of the Jacobian
    F_joint = J' * F_total';
    F1 = F_joint(1);
    F2 = F_joint(2);
    F3 = F_joint(3);
end

% Potential Field Gradient Visualization

[xGrid, yGrid, zGrid] = meshgrid(linspace(-25,25,10), linspace(-25,25,10), linspace(-25,25,10));

% Preallocate arrays for gradient components and magnitude
U_x = zeros(size(xGrid));
U_y = zeros(size(yGrid));
U_z = zeros(size(zGrid));
gradMag = zeros(size(zGrid));

% Use an attractive potential field centered at p_final.
k_att = 1.0;
p_final = [18, 9, 5];

for i = 1:numel(xGrid)
    % Compute potential U = 0.5 * k_att * norm(P - p_final)^2
    P = [xGrid(i), yGrid(i), zGrid(i)];
    U = 0.5 * k_att * norm(P - p_final)^2;
    
    % Compute numerical gradient (partial derivatives) using analytical form:
    grad = k_att * (P - p_final);
    U_x(i) = grad(1);
    U_y(i) = grad(2);
    U_z(i) = grad(3);
    
    gradMag(i) = norm(grad);
end

% Normalize vectors for better visualization
scale = 2; 
figure;
quiver3(xGrid, yGrid, zGrid, U_x, U_y, U_z, scale, 'LineWidth', 1.5);
colormap jet;
colorbar;
title('Potential Field Gradient Visualization');
xlabel('X-axis'); ylabel('Y-axis'); zlabel('Z-axis');
