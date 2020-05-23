function shapech = get_shape_change(ds)

L = size(ds, 2);
ds = padarray(ds, [0 30 0], 'both');
max_time = size(ds,3);
W = size(ds, 2);

iterations = 100;
for i = 1:iterations
    ds_ = mean(ds, 3);
    shift = zeros(max_time, 1);
    parfor t = 1:max_time
        shift(t) = find_shift(ds(:,:,t), ds_, W/2);
        ds(:,:,t) = circshift(ds(:,:,t), shift(t), 2);
    end

    power = mean(mean(abs(ds), 3), 1);
    [~, center] = max(smooth(power, 9));
    % [~, center] = max(power);
    ds = circshift(ds, W/2-center, 2);
    
    if all(shift == 0)
        break
    end
end

if ~all(shift == 0)
    fprintf('Failed to converge to solution.\n');
else
    fprintf('Converged in %d iterations.\n', i);
end

power = mean(mean(abs(ds), 3), 1);
power = arrayfun(@(i) sum(power(i:i+L-1)), 1:W-L+1);
[~, i] = max(power); best_win = i:i+L-1;
ds = ds(:,best_win,:);

S = svd(reshape(ds, [], size(ds,3)));
shapech = sum(S ./ max(S));

end

%% 

function k = find_shift(curr, ref, max_shift)

klist = -max_shift:max_shift;

r = nan(1, length(klist));
for i = 1:length(klist)
    shifted = circshift(curr, klist(i), 2);
    r(i) = corr(shifted(:), ref(:));
end

[~, i] = max(r);
k = klist(i);

end
