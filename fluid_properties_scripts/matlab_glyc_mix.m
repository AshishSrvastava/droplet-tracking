% modified following Andreas Volk's latest modifications
% this code takes VOLUME fraction as input - try Andreas's code if you
% prefer mass fraction
% This version 20 Dec 2017.

function[rho,eta]=density_viscosity_glycerine_mix(fraction_glyc,T)

volume_glycerol=fraction_glyc;
volume_water=1-fraction_glyc;

% Calculations:
total_volume=volume_glycerol+volume_water;
volume_fraction=volume_glycerol/total_volume;

%density_glycerol=1277-0.654*T;  % kg/m^3, equation 24
density_glycerol=1273.3-0.6121*T;  % UPDATED following Andreas Volkâ€™s suggestion

density_water=1000*(1-((abs(T-3.98))/615)^1.71);  % UPDATED following A.V.'s suggestion

mass_glycerol=density_glycerol*volume_glycerol; % kg
mass_water=density_water*volume_water; % kg
total_mass=mass_glycerol+mass_water; % kg
mass_fraction=mass_glycerol/total_mass;

viscosity_glycerol=0.001*12100*exp((-1233+T)*T/(9900+70*T)); % equation 22. Note factor of 0.001 -> converts to Ns/m^2
viscosity_water=0.001*1.790*exp((-1230-T)*T/(36100+360*T)); % equation 21. Again, note conversion to Ns/m^2

a=0.705-0.0017*T;
b=(4.9+0.036*T)*a^2.5;
alpha=1-mass_fraction+(a*b*mass_fraction*(1-mass_fraction))/(a*mass_fraction+b*(1-mass_fraction));
A=log(viscosity_water/viscosity_glycerol); % Note this is NATURAL LOG (ln), not base 10.
viscosity_mix=viscosity_glycerol*exp(A*alpha); % Ns/m^2, equation 6

% Andreas Volk polynomial:
c=1.78E-6*T.^2-1.82E-4*T+1.41E-2;
contraction=1+(c.*sin((mass_fraction).^1.31.*pi).^0.81);

density_mix=(density_glycerol*fraction_glyc+density_water*(1-fraction_glyc))*contraction; % equation 25

eta=viscosity_mix;
rho=density_mix;