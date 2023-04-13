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
    results = zeros(1, length(target_SNR_dB));
    
    for index = 1:length(target_SNR_dB)
        sum = 0;
        for n = 1:num_sim
            % Randomize the position of the user, and calulate the distances to each BS
            user = [rand(1)*radius rand(1)*radius];
            distances = zeros(1,num_stations);
            for bs = 1:num_stations
                distances(bs) = sqrt((base_stations(bs,1) - user(1))^2 + (base_stations(bs,2) - user(2))^2);
            end
    
            % Get the approximate SINR
            h = exprnd(1);
            numerator = h * distances(1)^(-alpha) * P;
            denominator = 0;
            for m = 2:length(distances)
                g = exprnd(1);
                denominator = denominator + (g * distances(m)^(-alpha) * P);
            end
            SINR = numerator / denominator;
            SINR_db = (10 * log10(SINR)) + N0;

            sum = sum + erf(SINR_db/target_SNR_dB(index));
        end
        results(index) = sum/num_sim;
    end
end