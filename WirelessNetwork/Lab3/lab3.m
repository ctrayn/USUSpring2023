%% Lab 3

radius = 1000;
P = 16; %dBw
N0 = -114; %dBm
alpha = 4;
reuse = 1/3;

target_SINR_db = linspace(-10, 20, 1000);
results = simuation(radius, N0, P, reuse, alpha, target_SINR_db);

figure();
plot(target_SINR_db, results);
xlabel("Target SINR db")
ylabel("Converge probability")