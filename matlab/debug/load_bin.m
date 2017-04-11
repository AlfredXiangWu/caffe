function dic = load_bin(fn)

NAME_LEN = 240;

dic = containers.Map;
fid = fopen(fn, 'rb');
while ~ feof(fid)
    name = fread(fid, NAME_LEN, '*char');
    if isempty(name)
        break;
    end
    name = deblank(name');
    disp(name);
    dims = fread(fid, 4, 'int');
    dims = fliplr(dims');
    disp(dims);
    count = prod(dims);
    data = fread(fid, count, 'single');
    data = reshape(data, dims);
    data = permute(data, [2 1 3 4]);
    dic(name) = data;
end
fclose(fid);