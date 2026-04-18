% Model         :  BLDC  six step commutation
% Description   :  Set Parameters for BLDC Six Step Control — BLY344S-240V-3000
% File name     :  mcb_bldc_sixstep_f28379d_data.m
% Modificado para motor Anaheim Automation BLY344S-240V-3000
% Parametros del hardware real de Axel (coronado25draft + script de init)
% Copyright 2020-2024 The MathWorks, Inc.

%% Set PWM Switching frequency
% CAMBIO: 20e3 Hz (Teknic) → 33.3 kHz (hardware real de Axel, Ts=3e-5)
PWM_frequency   = 1/3e-5;          %Hz     // converter s/w freq
T_pwm           = 1/PWM_frequency;  %s      // PWM switching time period
Ts_motor_simscape = T_pwm/100;

%% Set Sample Times
Ts          	= T_pwm;            %sec    // Sample time step for controller
Ts_simulink     = T_pwm/2;          %sec    // Simulation time step for model simulation
Ts_motor        = T_pwm/2;          %Sec    // Simulation sample time
Ts_inverter     = T_pwm/2;          %sec    // Simulation time step for average value inverter
Ts_speed        = 10*Ts;            %Sec    // Sample time for speed controller

%% Set data type for controller & code-gen
dataType = 'single';                % Floating point code-generation

%% System Parameters — Motor BLY344S-240V-3000 (Anaheim Automation)
% Cargamos estructura base de MCB y sobreescribimos con parametros reales
bldc = mcb.getPMSMParameters('Teknic2310P');  % estructura base

% Identificacion
bldc.model  = 'Anaheim-BLY344S-240V-3000';
bldc.sn     = '002';

% Parametros electricos y mecanicos (fuente: script Axel + coronado25draft)
bldc.p          = 4;            %           // Pole pairs
bldc.Rs         = 1.1;          %Ohm        // Stator resistance
bldc.Ld         = 0.00372;      %H          // D-axis inductance
bldc.Lq         = 0.00372;      %H          // Q-axis inductance (motor superficie, Ld=Lq)
bldc.J          = 2.399e-4;     %kg·m²      // Rotor inertia
bldc.B          = 0.0006738;    %Nm·s       // Viscous friction
bldc.Ke         = 19.443;       %Vpk_LL/krpm// BEMF constant (convencion MCB)
bldc.Kt         = 0.1865;       %Nm/A       // Torque constant
bldc.I_rated    = 3.18;         %A          // Rated phase current (peak)
bldc.N_max      = 3000;         %rpm        // Max speed

% Encoder AMT333S-V @ 2048 PPR (coronado25draft, Fig.7)
bldc.QEPSlits   = 2048;

% Offset de posicion (del script de Axel)
bldc.PositionOffset = 0.1364;   %PU

% FluxPM y T_rated — derivados de Ke y I_rated (formula MCB estandar)
bldc.FluxPM  = (bldc.Ke) / (sqrt(3) * 2*pi * 1000 * bldc.p / 60);  %Wb
bldc.T_rated = (3/2) * bldc.p * bldc.FluxPM * bldc.I_rated;          %Nm

%% Hall Sequence
% NOTA: secuencia pendiente de calibracion con hardware real.
% Usar workflow mcb_hall_calibration_f28379d para obtener la correcta.
% Por ahora se mantiene la misma que Teknic como placeholder.
bldc.HallSequence = [4,6,2,3,1,5]; % *** VERIFICAR CON HARDWARE ***

%% Target & inverter parameters
% Target: TI F28379D LaunchPad (mismo hardware que Axel)
target = mcb.getProcessorParameters('F28379D', PWM_frequency);
target.comport = '<Select a port...>';

% Inversor: BoostXL-DRV8305
% NOTA: Axel usa TMDSHVMTRPFCKIT (HV Kit, Vdc=200V).
% BoostXL es el inversor del ejemplo MCB. Para simulacion es suficiente.
% Si se despliega en hardware, cambiar a los parametros del HV Kit.
inverter = mcb.getInverterParameters('BoostXL-DRV8305');
inverter.CtSensCOffset = 2283;  % del script de Axel (CtSensCOffset real)

%% Inverter Calibration
inverter.ADCOffsetCalibEnable = 1; % Enable automatic ADC offset calibration

%% Actualizar parametros del inversor segun motor y target
inverter = mcb.updateInverterParameters(bldc, inverter, target);

% Limites de offset ADC
inverter.CtSensOffsetMax = 2500;
inverter.CtSensOffsetMin = 1500;

%% Derive Characteristics
bldc.N_base = mcb.getMotorBaseSpeed(bldc, inverter); %rpm

%% PU System
PU_System = mcb.getPUSystemParameters(bldc, inverter);

%% Controller design
PI_params = mcb.getPIControllerParameters(bldc, inverter, PU_System, ...
            T_pwm, 2*Ts, Ts_speed);
PI_params.delay_Currents = 1;
PI_params.delay_Position = 1;

%% Verificacion en consola
fprintf('\n=== BLY344S-240V-3000 — Parametros derivados ===\n')
fprintf('FluxPM    : %.4f Wb\n',  bldc.FluxPM)
fprintf('T_rated   : %.4f Nm\n',  bldc.T_rated)
fprintf('N_base    : %.1f rpm\n', bldc.N_base)
fprintf('Ts        : %.1f us\n',  Ts*1e6)
fprintf('PWM freq  : %.1f kHz\n', PWM_frequency/1e3)
fprintf('=================================================\n\n')

disp(bldc);
disp(inverter);
disp(target);
disp(PU_System);