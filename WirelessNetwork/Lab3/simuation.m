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
    
    num_sim = 3000;
    results = zeros(1, length(target_SNR_dB));
    
    for index = 1:length(target_SNR_dB)
        sum_of_sims = 0;
        for n = 1:num_sim
            % Randomize the position of the user, and calulate the distances to each BS
            r = rand(1)*radius;
            theta = rand(1)*2*pi;
            user = [r*cos(theta) r*sin(theta)];
            distances = zeros(1,num_stations);
            for bs = 1:num_stations
                distances(bs) = sqrt((base_stations(bs,1) - user(1))^2 + (base_stations(bs,2) - user(2))^2);
            end
    
            % Get the approximate SINR
            h = exprnd(1);
            numerator = h * distances(1)^(-alpha) * P;
            g = exprnd(1);
            denominator = (g .* distances(2:end).^(-alpha) .* P);
            SINR = numerator / sum(denominator) + N0;
            %SINR_db = (10 * log10(SINR)) + N0;

            sum_of_sims = sum_of_sims + erf(SINR/target_SNR_dB(index));
        end
        results(index) = sum_of_sims/num_sim;
    end
end