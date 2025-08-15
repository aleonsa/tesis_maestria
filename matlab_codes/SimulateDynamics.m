function [x1dot, x2dot] = SimulateDynamics(theta1, theta2, u)

    x1dot = x2;
    x2dot = -theta1 * sin(x1) + theta2 * u;

end

