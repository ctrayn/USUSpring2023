%% Simulating with given factors

function results = simuation(radius, reuse, N0, P, alpha, target_SNR_dB)
    num_stations = 7;
    base_stations = get_base_stations(radius, reuse);
    disp(base_stations)
    
    num_sim = 5000;
    results = zeros(1, length(target_SNR_dB));
    
    for index = 1:length(target_SNR_dB)
        %sum_of_sims = 0;
        num_above = 0;
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
            denominator = sum(g .* distances(2:end).^(-alpha) .* P) + N0;
            %disp(denominator)
            SINR = numerator / denominator;
            %disp(SINR)
            SINR_dB = (10 * log10(SINR)); % Convert to dB

            if SINR_dB >= target_SNR_dB(index)
                num_above = num_above + 1;
            end

            %sum_of_sims = sum_of_sims + erf(SINR/target_SNR_dB(index));
        end
        results(index) = num_above/num_sim;
    end
end