%% Lab 3

P = 10^((16 + 30)/10); %dBw to decimal
N0 = 10^((-114 - 30)/10); %dBm to decimal
target_SINR_db = linspace(-10, 20, 50);

%% Simulation 1
alpha = 4;
reuse = 1/3;

radius = 1000;
results1 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);

radius = 5000;
results2 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);

plot_results(results1, results2, target_SINR_db, ["Radius=1000","Radius=5000"], "Simulation 1");

%% Simulation 2

radius = 1000;
reuse = 1/3;

alpha = 2.5;
results1 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);

alpha = 4;
results2 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);

plot_results(results1, results2, target_SINR_db, ["Alpha=2.5","Alpha=4"], "Simulation 2");

%% Simulation 3

radius = 1000;
alpha = 4;

reuse = 1/3;
results1 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);

reuse = 1;
results2 = simuation(radius, reuse, N0, P, alpha, target_SINR_db);


plot_results(results1, results2, target_SINR_db, ["Reuse=1/3","Reuse=1"], "Simulation 3");

