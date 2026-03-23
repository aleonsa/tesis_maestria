%% Set PWM Switching frequency
PWM_frequency 	= 1/(3e-5);%31.25e3;%1/(2.7e-5);%40e3;    %Hz          // converter s/w freq
T_pwm           = 1/PWM_frequency;  %s  // PWM switching time period

%% Set Sample Times simulation
Ts          	= T_pwm;        %sec        // simulation time step for controller
Ts_speed        = 10*1*Ts;        %Sec        // Sample time for speed controller
Ts_com=2*1*Ts;
%Ts_com=Ts;
%Ts_speed        = 0.0005; 
Tsfsmpc=Ts;
Tsob=Ts;
%% Set data type for controller & code-gen
dataType = 'single';            % Floating point code-generation, Fixed point is not supported.
target = mcb_SetProcessorDetails('F28379D',PWM_frequency);
%% System Parameters // Hardware parameters Motor
        pmsm.model  = 'Anaheim-BLY34S-240V-3000';%  // Manufacturer Model Number
        pmsm.sn     = '002';            %           // Manufacturer Model Number
        pmsm.p      = 4;                %           // Pole Pairs for the motor
        pmsm.Rs     = 1.1;              %Ohm        // Stator Resistor
        pmsm.Ld     = 0.00372;           %H          // D-axis inductance value
        pmsm.Lq     = 0.00372;           %H          // Q-axis inductance value
%          pmsm.Ld     = 9e-3;           %H          // D-axis inductance value
%         pmsm.Lq     = 9e-3;           %H          // Q-axis inductance value
%%%%%%%%%%%%%%%%%%%%
%         pmsm.J      = 2.7948e-04; %Kg-m2      // Inertia in SI units
%         pmsm.B      =  0.0006738; %Kg-m2/s    // Friction Co-efficient
%         pmsm.Ke     = 42.26;                %Bemf Const // Vpk_LL/krpm
%         pmsm.Kt     = 0.659972;          %Nm/A       // Torque constant
        % pmsm.J      = 2.4795e-04; %Kg-m2  
        pmsm.J      = 2.399e-4;%2.9795e-04; %Kg-m2 // Inertia in SI units
       pmsm.B      = 0.0006738; %Kg-m2/s    // Friction Co-efficient
       pmsm.B      = 0; %Kg-m2/s
        % pmsm.Ke     = 39.1547;                %Bemf Const // Vpk_LL/krpm
         pmsm.Ke     = 19.443;                %Bemf Const // Vpk_LL/krpm
       pmsm.Kt     =0.1865; 
%pmsm.Kt     =0.3739;          %Nm/A       // Torque constant
 %%%%%%%%%%%%%%%
        pmsm.I_rated= 3.18;              %A          // Rated current (phase-peak)
        pmsm.N_max  = 3000;            %rpm        // Max speed
        pmsm.PositionOffset = 0.1364;	%PU position// Position Offset
        pmsm.QEPSlits       = 2000;     %           // QEP Encoder Slits
        pmsm.FluxPM     = (pmsm.Ke)/(sqrt(3)*2*pi*1000*pmsm.p/60); %PM flux computed from Ke
        pmsm.T_rated    = (3/2)*pmsm.p*pmsm.FluxPM*pmsm.I_rated;   %Get T_rated from I_rated
  %% System Parameters Inverter
       inverter.model         = 'HVKit'; 	% 		// Manufacturer Model Number
       inverter.sn            = 'INV_XXXX';         	% 		// Manufacturer Serial Number
       inverter.V_dc          = 200;       				%V      // DC Link Voltage of the Inverter
            %Note: inverter.I_max = 1.65V/(Rshunt*10V/V) for 3.3V ADC with offset of 1.65V
            %This is modified to match 3V ADC with 1.65V offset value for %LaunchXL-F28379D
            inverter.I_max         = 10*((3-1.65)/1.65);  %Amps   // Max current that can be measured by 3.0V ADC
            inverter.I_trip        = 9.7 ;       				%Amps   // Max current for trip
            inverter.Rds_on        = 2e-3;     				%Ohms   // Rds ON for BoostXL-DRV8305
            inverter.Rshunt        = 0.02;    				%Ohms   // Rshunt for BoostXL-DRV8305
            inverter.MaxADCCnt     = 4095;     				%Counts // ADC Counts Max Value
            inverter.CtSensAOffset =2284; %2250;        			%Counts // ADC Offset for phase-A 
            %inverter.CtSensAOffset = 2249;   
            inverter.CtSensBOffset = 2239;   % 2248;  
            %inverter.CtSensBOffset = 2244;   % 2248;  
            inverter.CtSensCOffset = 2283;     
            inverter.VtSensUOffset = 0;        			%Counts // ADC Offset for phase-A 
            inverter.VtSensVOffset = 0;   % 2248;  
            inverter.VtSensWOffset = 0;  
            %Counts // ADC Offset for phase-A 
            inverter.ADCGain       = 1;                     %       // ADC Gain factor scaled by SPI
			inverter.EnableLogic   = 1;    					% 		//Active high for DRV8305 enable pin (EN_GATE)
			inverter.invertingAmp  = 1;   					% 		//Non inverting current measurement amplifier  
            inverter.R_board= 1e-4;
            inverter.ISenseMax     =9.0909;
            m = 2*pi;
Offset=0;%rad
Tscomms=2*Ts*600;
%Tscomms=Ts*600; %%%FOC PCH
%Tscomms=6e-5*600; %%%FOC
%Tscomms=2*Ts*600; %%%FOC
%X0=[5*pi/3,pi,4*pi/3,pi/3,0,2*pi/3];
% X0=[0.9513,4.853,6.022,2.915,1.696,4.109];
% Xof(1)=mod(X0(1)+Offset,m);
% Xof(2)=mod(X0(2)+Offset,m);
% Xof(3)=mod(X0(3)+Offset,m);
% Xof(4)=mod(X0(4)+Offset,m);
% Xof(5)=mod(X0(5)+Offset,m);
% Xof(6)=mod(X0(6)+Offset,m);            
         %% Derive Characteristics
pmsm.N_base = mcb_getBaseSpeed(pmsm,inverter); %rpm // Base speed of motor at given Vdc

% mcb_getCharacteristics(pmsm,inverter);

%% PU System details // Set base values for pu conversion

SI_System = mcb_SetSISystem(pmsm);

%% Controller design // Get ballpark values!

 PI_params = mcb.internal.SetControllerParameters(pmsm,inverter,SI_System,T_pwm,Ts,Ts_speed);

%Updating delays for simulation
% PI_params.delay_Currents    = int32(Ts/Ts_simulink);
% PI_params.delay_Position    = int32(Ts/Ts_simulink);
% PI_params.delay_Speed       = int32(Ts_speed/Ts_simulink);
% PI_params.delay_Speed1      = (PI_params.delay_IIR + 0.5*Ts)/Ts_speed;   

% PI_params.Kp_id=100;
% PI_params.Kp_i=100;
% PI_params.Ki_id=0;
% PI_params.Ki_i=0;

%PI_params.Kp_speed=0.006;
%PI_params.Ki_speed=0.008;

% %% Parametros microcontrolador // 
%         Ts=single(T_pwm);
%         R=single(1) ;
%        %L=single(1.4e-3);
%        %L=single(4.5e-3);
%         L=single(2.01e-3);
%       % L=single(1.4e-3);
%         M=(3/7)*L;
%         Ls=single(L+M);
%         gmax=single(1000);
%         ud=single(170);
%         MaxValue=1e8;
% 
% %%%%%%%%FSMPC sobremodulado
% % 
% Div=single(3); %%%%Numero de divisiones 
% dy0=single(0.15);
% dyV=(1-dy0)/Div;
% Tv=dyV*Ts;
% Tvz=Ts*(dy0/2+dyV);


%%%%%%%%%FSM2PC%%%%%%%%%
% 
% Div=single(4); %%%%Numero de divisiones 
% dy0=single(0.15);
% dyV=(1-dy0);
% 
% Tab=single([0 0;
%     0.25 0;
%     0 0.25;
%     0.5 0;
%     0.25 0.25
%     0 0.5;
%     0.75 0;
%     0.5 0.25;
%     0.25 0.5;
%     0 0.75;
%     1 0;
%     0.75 0.25
%     0.5 0.5
%     0.25 0.75
%     0 1]);

% %%%%%%%%%%%%%FSMPC3V SImple 15%%%%%%%%%
n=single(21);
are=single(0);
ar=single(round(1/10,3));

    for j= 1:n  

if j<=20
v1(j)=single(1-mod(j,2));
v(j)=abs(single(0-mod(j,2)));
are=are+rem(j, 2);
Tab(1,j)=v(j)*ar*are;
Tab(2,j)=v1(j)*ar*are;

else
v1(j)=single(0);
v(j)=single(0);
Tab(1,j)=v(j);
Tab(2,j)=v1(j);
end
    end
Tabf=round(Tab',3);

% %%%%%%%%FSMPC3V%%%%%%%%%
% n=single(11);
% ar=single(round(0.85/7,3));
% 
%     for j= 1:n  
% 
% if j<=10
% v1(j)=single(1-mod(j,2));
% v(j)=abs(single(0-mod(j,2)));
% % elseif j==7
% % v1(j)=single(0);
% % v(j)=single(0);
% else
% v1(j)=single(0);
% v(j)=single(0);
% %  v1(j)=single(1-mod(j,2));
% % v(j)=abs(single(0-mod(j,2)));
% end
% 
% if j<=2
%   Tab(1,j)=v(j)*ar*single(2);
%   Tab(2,j)=v1(j)*ar*single(2);
% elseif j<=4
%   Tab(1,j)=ar*v(j)*single(4);
%   Tab(2,j)=ar*v1(j)*single(4);
% %   elseif j<=6
% %   Tr1=ar*single(6);
% elseif j<=6
%     % Tr1=ar*single(8);
%       Tab(1,j)=ar*v(j)*single(6);
%        Tab(2,j)=ar*v1(j)*single(6);
% else
%        Tab(1,j)=v(j);
%        Tab(2,j)=v1(j);
% end
%     end
% Tabf=round(Tab',3);
%% Parametros microcontrolador observador // 
%J=single(2.7948e-04);
% J=single(2.3998e-4); %Solo motor.
% d = single(0.0006738);
% Lf=single(7000);
% Tco=single(0.06768);
% A=[0,1;0,-d/J] ;
% B=[0;1];
% D=[0;1/J];
% C=[1 0];
% Rk=0.4;
% Q=1*[0.1,0;0,0.8] 
% %Q=1e-3*eye(2);
% [X1,K1,L1] = care(A,C',Q,Rk);
% Lo=X1*C'*Rk^-1;
% % q = [-100,-200];
% % Lo = place(A',C',q)';
% %Lo=[10,10.6148]';
% syms s  
% dd=(C*A*D);
% H=det(A-Lo*C-s*eye(2));
% s=0;
% a2=single(-(Lo(1)+d/J));
% a1=single(-(Lo(2)+Lo(1)*d/J));
% %k2 =single( 2*(Lf^(1/3)));%%%Parece ser 2
% k2 =single( 2*(Lf^(1/3)));%%%Parece ser 2
% k1= single(2.12*(Lf^(2/3)));
% %k11= single(1.5*(Lf^(1/2)));
% k0 =single( 1.1*Lf);
% %Tsob=2.5e-5;
% Kop=inv([C;C*(A-L*C)]);
% 
% Ls=single(0.0037);