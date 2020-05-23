function gainch = get_gain_change(ds)

gainch = std(std(reshape(ds, [], size(ds, 3)), [], 1), [], 2);

end
