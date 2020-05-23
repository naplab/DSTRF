function tmphld = get_temporal_hold(ds)

max_lookahead = 30;
ds = padarray(ds, [0 max_lookahead 0], 0, 'post');
max_time = size(ds, 3) - max_lookahead;

% shift all values, find best
pval = nan(max_lookahead, 1);
parfor lag = 1:max_lookahead
    dist0 = nan(max_time, 1);
    dist1 = nan(max_time, 1);
    for t = 1:max_time
        ref = ds(:, :, t); ref = ref(:);
        target = ds(:, :, t+lag);
        
        [dist0(t), ~] = corr(ref, target(:), 'type', 'Pearson');
        
        aligned = unshift(target, lag);
        [dist1(t), ~] = corr(ref, aligned(:), 'type', 'Pearson');
    end
    
    pval(lag) = signrank(dist0, dist1, 'tail', 'left');
end

tmphld = find([pval; 1] >= 0.001, 1);

end

%% 

function x = unshift(x, k)

x = padarray(x(:, 1:end-k), [0 k], 0, 'pre');

end
