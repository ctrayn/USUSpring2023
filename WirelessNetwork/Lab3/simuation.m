%% Simulating with given factors

function results = simuation(radius, N0, P, reuse, alpha, target_SNR_dB)
    num_stations = 7;
    base_stations = [
        [0 0] 
        [ 1.5*radius  0.5*sqrt(3)*radius]
        [ 1.5*radius -0.5*sqrt(3)*radius]
        [-1.5*radius  0.5*sqrt(3)*radius]
        [-1.5*radius -0.5*sqrt(3)*radius]
        [0  radius*sqrt(3)]
        [0 -radius*sqrt(3)]
    ];
    
    num_sim = 5000;
    results = zeros(1,num_sim);
    
    for n = 1:num_sim
        user = [rand(1)*radius rand(1)*radius];
        distances = zeros(1,num_stations);
        for index = 1:num_stations
            distances(index) = sqrt((base_stations(index,1) - user(1))^2 + (base_stations(index,2) - user(2))^2);
        end

        h = exprnd(1);
        numerator = h * distances(1)^(-alpha) * P;
        denominator = 0;
        for m = 2:length(distances)
            g = exprnd(1);
            denominator = denominator + (g * distances(m)^(-alpha) * P);
        end
        SINR = numerator / (denominator + N0);
        SINR_db = 10 * log10(SINR);
    end
end