clc;
clear;
close all;

%% Loading
% Path to .mhd file
mhd_file = "D:\tested\detected\1.3.6.1.4.1.14519.5.2.1.6279.6001.206539885154775002929031534291.mhd";

% Extraction of the filename without path and extension
[~, name, ~] = fileparts(mhd_file);  % 'name' will be the dot separated number sequence

% Reading the CSV file
opts = detectImportOptions("D:\annotations.csv");
opts = setvartype(opts, {'coordX','coordY','coordZ','diameter_mm'}, 'double');
data = readtable("D:\annotations.csv", opts);

% Matching the filename (seriesuid) with the CSV's first column
matches = strcmp(strtrim(data.seriesuid), name);

no=0;
% Extraction of values if match found
% Nodule coordinates (world) and diameter
if any(matches)
    coordX = data.coordX(matches);
    coordY = data.coordY(matches);
    coordZ = data.coordZ(matches);
    diameter_mm  = data.diameter_mm(matches);

    % % Display the results
    % disp('Match found:');
    % disp(table(coordX, coordY, coordZ, diameter_mm));
else
    disp(['No matching entry found for ', name, ' in annotations.csv.']);
    no=1;
end

% Picking mhd slice
if no==0
    i=1;
    coordX = coordX(i);
    coordY = coordY(i);
    coordZ = coordZ(i);
    diameter_mm  = diameter_mm(i);
end

% Read metadata from .mhd file
fid = fopen(mhd_file, 'r');
if fid == -1
    error('Cannot open .mhd file: %s', mhd_file);
end
metadata = textscan(fid, '%s %s', 'Delimiter', '=', 'CommentStyle', '#');
fclose(fid);

% Extract dimensions, raw file, offset, and element spacing
dims = [];
raw_file = '';
offset = [];
spacing = [];
for i = 1:length(metadata{1})
    key = strtrim(metadata{1}{i});
    value = strtrim(metadata{2}{i});
    if strcmp(key, 'DimSize')
        dims = str2num(value);
    elseif strcmp(key, 'ElementDataFile')
        raw_file = value;
    elseif strcmp(key, 'Offset')
        offset = str2num(value);
    elseif strcmp(key, 'ElementSpacing')
        spacing = str2num(value);
    end
end

% Validate and display metadata
if isempty(dims) || isempty(raw_file) || isempty(offset) || isempty(spacing)
    error('Missing metadata in .mhd file (DimSize, ElementDataFile, Offset, or ElementSpacing).');
end

% disp('Metadata from .mhd file:');
% disp(['DimSize: ', num2str(dims)]);
% disp(['Offset: ', num2str(offset)]);
% disp(['ElementSpacing: ', num2str(spacing)]);

% Read and reshape raw data
raw_file_path = fullfile(fileparts(mhd_file), raw_file);
fid = fopen(raw_file_path, 'r');
if fid == -1
    error('Cannot open .raw file: %s', raw_file_path);
end
img_data = fread(fid, prod(dims), 'int16'); % Change to 'uint8' if ElementType is MET_UCHAR
fclose(fid);
img_data = reshape(img_data, dims([2, 1, 3]));
img_data = permute(img_data, [2, 1, 3]);

% Convert world coordinates to voxel coordinates
if no==0
    voxelX = (coordX - offset(1)) / spacing(1);
    voxelY = (coordY - offset(2)) / spacing(2);
    voxelZ = (coordZ - offset(3)) / spacing(3);

    % Debug: Display calculated voxel coordinates
    % disp('Calculated voxel coordinates:');
    % disp(['voxelX: ', num2str(voxelX)]);
    % disp(['voxelY: ', num2str(voxelY)]);
    % disp(['voxelZ: ', num2str(voxelZ)]);
    % disp(['Nodule Diameter (mm): ', num2str(diameter_mm)]);
end

% Select slice closest to nodule's z-coordinate
if no==0
    slice_index = round(voxelZ); 
else
    slice_index = 60;
end

if slice_index < 1 || slice_index > size(img_data, 3)
    error('Nodule z-coordinate (%.2f mm) is outside the volume (z-range: %.2f to %.2f mm).', ...
        coordZ, offset(3), offset(3) + (size(img_data, 3) - 1) * spacing(3));
end

%% Extract 2D slice
slice_2d = img_data(:, :, slice_index);

    % Display image without red mark
    figure('Name', 'Lung Slice Without Nodule Mark');
    imshow(slice_2d, [-1000, 400]); % Lung window
    colormap gray;
    %title(sprintf('Lung Slice %d (z=%.2f mm)', slice_index, coordZ));
    axis off;

if no == 0
    % Display image with red mark
    figure('Name', 'Lung Slice With Nodule Mark');
    imshow(slice_2d, [-1000, 400]); % Lung window
    colormap gray;
    hold on;
    % Plot nodule as a red circle
    radius_voxels = diameter_mm / (2 * spacing(1)); % Radius in voxels (assuming isotropic x-y spacing)
    theta = 0:0.01:2*pi;
    x_circle = voxelX + radius_voxels * cos(theta);
    y_circle = voxelY + radius_voxels * sin(theta);
    plot(x_circle, y_circle, 'r-', 'LineWidth', 2);
    % Add text label for coordinates
    text(voxelX + radius_voxels + 10, voxelY, sprintf('X: %.2f\nY: %.2f', voxelX, voxelY), ...
        'Color', 'cyan', 'FontSize', 8);
    title(sprintf('Lung Slice %d with Nodule (z=%.2f mm)', slice_index, coordZ));
    axis off;
end

%% Normalizing

% ✅ CT-style Normalization to [0, 1] with Lung Window
% Step 1: Clip intensities to [-1000, 400]
slice_clipped = slice_2d;
slice_clipped(slice_clipped < -1000) = -1000;
slice_clipped(slice_clipped > 400) = 400;

% Step 2: Normalize clipped image to [0, 1]
norm_slice = (slice_clipped + 1000) / (1400);  % range = 400 - (-1000) = 1400

% Display Normalized Slice
figure('Name', '✅ Normalized CT Slice [0, 1]');
imshow(norm_slice);
colormap gray;
title('✅ Normalized CT Slice [0, 1] with Lung Window');
axis off;

min_val = min(norm_slice(:));
max_val = max(norm_slice(:));

disp('Checking Normalization Numerically ')

disp(['Min: ', num2str(min_val), '   Max: ', num2str(max_val)]);


%% Denoising

% Pad the image to handle borders
padded = padarray(norm_slice, [1 1], 'symmetric');
[rows, cols] = size(norm_slice);
denoised_manual = zeros(size(norm_slice));

% Apply 3x3 median filter manually
for i = 2:rows+1
    for j = 2:cols+1
        window = padded(i-1:i+1, j-1:j+1);
        denoised_manual(i-1, j-1) = median(window(:));
    end
end

% Display
figure('Name', '✅ Manual Denoised CT Slice');
imshow(denoised_manual);
colormap gray;
title('✅ Denoised with Manual 3x3 Median Filter');
axis off;

figure;
subplot(1,2,1);
histogram(norm_slice(:), 100); 
title('Before Median Filter');
xlabel('Pixel Intensity');
ylabel('Frequency');

subplot(1,2,2);
histogram(denoised_manual(:), 100); 
title('After Median Filter');
xlabel('Pixel Intensity');
ylabel('Frequency');


%% Adaptive histogram equalization

%adhist_eq = adapthisteq(img_denoised_man);

  function output = manual_adapthisteq(img, tileSize, clipLimit)
% Fully manual Adaptive Histogram Equalization (CLAHE-style)
% img: normalized grayscale [0,1]
% tileSize: scalar (e.g., 64)
% clipLimit: relative histogram clip value (e.g., 0.01)

% Step 0: Setup
[rows, cols] = size(img);
levels = 256;
img_q = floor(img * (levels - 1));
nTilesY = ceil(rows / tileSize);
nTilesX = ceil(cols / tileSize);
cdf_map = cell(nTilesY, nTilesX);

% Step 1: Compute local histograms and CDFs per tile
for i = 1:nTilesY
    for j = 1:nTilesX
        % Get tile indices
        r_start = (i-1)*tileSize + 1;
        r_end   = min(i*tileSize, rows);
        c_start = (j-1)*tileSize + 1;
        c_end   = min(j*tileSize, cols);
        tile = img_q(r_start:r_end, c_start:c_end);

        % Manual histogram
        hist_tile = zeros(1, levels);
        for v = 0:levels-1
            hist_tile(v+1) = sum(tile(:) == v);
        end

        % CLAHE-style clip (limit extreme bins)
        maxCount = clipLimit * numel(tile);
        excess = sum(max(0, hist_tile - maxCount));
        hist_tile = min(hist_tile, maxCount);
        hist_tile = hist_tile + excess / levels;

        % Normalize and get CDF
        cdf = cumsum(hist_tile) / sum(hist_tile);
        cdf_map{i,j} = cdf;
    end
end

% Step 2: Interpolate pixel intensities
output = zeros(size(img));
for y = 1:rows
    for x = 1:cols
        % Get tile indices
        ty = min(floor((y-1)/tileSize) + 1, nTilesY);
        tx = min(floor((x-1)/tileSize) + 1, nTilesX);

        % Get position inside tile
        y1 = (ty - 1)*tileSize + 1;
        x1 = (tx - 1)*tileSize + 1;
        dy = min((y - y1) / tileSize, 1);
        dx = min((x - x1) / tileSize, 1);

        % Gather 4 neighboring CDFs
        cdf00 = cdf_map{ty, tx};
        cdf10 = cdf_map{min(ty+1, nTilesY), tx};
        cdf01 = cdf_map{ty, min(tx+1, nTilesX)};
        cdf11 = cdf_map{min(ty+1, nTilesY), min(tx+1, nTilesX)};

        % Interpolate CDF value for pixel intensity
        val = img_q(y,x)+1;
        cdf_interp = (1-dy)*(1-dx)*cdf00(val) + ...
                     dy*(1-dx)*cdf10(val) + ...
                     (1-dy)*dx*cdf01(val) + ...
                     dy*dx*cdf11(val);

        output(y,x) = cdf_interp;
    end
end
end


% Apply manual AHE
enhanced = manual_adapthisteq(denoised_manual, 64, 0.01);

figure;
imshow(enhanced, []);
title('Manual CLAHE Implementation');

 %figure(5)

%imshow(adhist_eq); 
%title(' Adaptive Histogram Equalized Image'); 

%disp('-> Adaptive Histogram Equalization Done ')

% ✅ Adaptive Histogram Equalization Checking 

figure;

subplot(1,2,1);
histogram(denoised_manual(:), 100);  % Before AHE
title('Before AHE');
xlabel('Pixel Value');

subplot(1,2,2);
histogram(enhanced(:), 100);  % After AHE
title('After AHE');
xlabel('Pixel Value');

figure;
histogram(norm_slice(:), 100);  % 100 bins gives fine detail
title('Pixel Intensity Histogram of Normalized CT Slice');
xlabel('Pixel Value');
ylabel('Frequency');

%% Lung Masking – Equivalent of kmader’s Python logic

% Input: enhanced → CLAHE-processed normalized slice
% Output: lung_mask, masked_lung_img, left_lung, right_lung

% Step 1: Otsu threshold to isolate lungs (lungs = darker regions)
otsu_level = graythresh(enhanced);
binary_otsu = enhanced < otsu_level;

% Step 2: Remove small border-connected junk
binary_cleaned = imclearborder(binary_otsu);

% Step 3: Fill holes inside lungs
binary_filled = imfill(binary_cleaned, 'holes');

% Step 4: Morphological opening and closing
se = strel('disk', 3);
binary_opened = imopen(binary_filled, se);
binary_closed = imclose(binary_opened, strel('disk', 1));

% Step 5: Identify and separate the two largest connected components (lungs)
cc = bwconncomp(binary_closed);
stats = regionprops(cc, 'Area');
[~, idx] = sort([stats.Area], 'descend');

% Initialize individual lung masks
left_lung = false(size(binary_closed));
right_lung = false(size(binary_closed));

% Extract the two largest components separately
if numel(idx) >= 1
    left_lung(cc.PixelIdxList{idx(1)}) = true;
end
if numel(idx) >= 2
    right_lung(cc.PixelIdxList{idx(2)}) = true;
end

% Step 6: Apply individual cleanup to each lung
% Clean up left lung
if any(left_lung(:))
    left_lung = imclose(left_lung, strel('disk', 25));
    left_lung = imfill(left_lung, 'holes');
    % Additional smoothing for boundary irregularities
    left_lung = imopen(left_lung, strel('disk', 2));
end

% Clean up right lung
if any(right_lung(:))
    right_lung = imclose(right_lung, strel('disk', 25));
    right_lung = imfill(right_lung, 'holes');
    % Additional smoothing for boundary irregularities
    right_lung = imopen(right_lung, strel('disk', 2));
end

% Step 7: Combine cleaned lungs using OR operation
lung_mask = left_lung | right_lung;

% Optional: Final combined mask refinement (if needed)
% lung_mask = imclose(lung_mask, strel('disk', 2));

% Step 8: Apply the lung mask to the enhanced image
masked_lung_img = enhanced;
masked_lung_img(~lung_mask) = 0;

% Optional: Create individual masked images for analysis
left_lung_img = enhanced;
left_lung_img(~left_lung) = 0;

right_lung_img = enhanced;
right_lung_img(~right_lung) = 0;

% % Quality check and statistics
% fprintf('Lung segmentation completed:\n');
% fprintf('- Left lung area: %d pixels\n', sum(left_lung(:)));
% fprintf('- Right lung area: %d pixels\n', sum(right_lung(:)));
% fprintf('- Total lung area: %d pixels\n', sum(lung_mask(:)));
% fprintf('- Lung area ratio (L/R): %.2f\n', sum(left_lung(:)) / max(sum(right_lung(:)), 1));

% Optional visualization
if exist('enhanced', 'var')
    figure;
    subplot(2,3,1); imshow(enhanced, []); title('Enhanced Image (CLAHE)');
    subplot(2,3,2); imshow(binary_otsu); title('Otsu Mask');
    subplot(2,3,3); imshow(binary_cleaned); title('Cleaned Border');
    subplot(2,3,4); imshow(binary_closed); title('Post Morphology');
    subplot(2,3,5); imshow(lung_mask); title('Final Lung Mask');
    subplot(2,3,6); imshow(masked_lung_img, []); title('Masked Lung Output');
    sgtitle('✅ Lung Segmentation Based on kmader (Kaggle) Logic');
end

%imwrite(masked_lung_img,'s1.png');

%% ✅ Enhanced Nodule Detection via Edge Filtering and Thresholding

% Step 1: Apply Sobel filter to enhanced lung image
sobel_x = fspecial('sobel');
sobel_y = sobel_x';

grad_x = imfilter(masked_lung_img, sobel_x, 'replicate');
grad_y = imfilter(masked_lung_img, sobel_y, 'replicate');
grad_mag = sqrt(grad_x.^2 + grad_y.^2);

% Step 2: Normalize gradient
grad_mag = mat2gray(grad_mag);

% Step 3: Apply custom threshold (better control than Otsu)
t = graythresh(grad_mag);
disp(t);
thresh_val = t*2.5;  % adjust between 0.2 - 0.4 if needed
edges_binary = grad_mag > thresh_val;

% Step 4: Restrict to lung region only
edges_binary(~lung_mask) = 0;

% Step 7: Display Results
figure(100);
subplot(2,2,1); imshow(grad_mag, []); title('Sobel Gradient');
subplot(2,2,2); imshow(edges_binary); title('Edges (Post Thresholding)');

area_mm2 = pi * (6 / 2)^2;
Ap2min = area_mm2 / (spacing(1) * spacing(2));
Ap2min = round(Ap2min);

%figure(9);
% Step 5: Morphological cleanup
edges_binary = bwareaopen(edges_binary, 30);  % remove tiny blobs
%subplot(2,1,1); imshow(edges_binary); title('areaopen');
edges_binary = imclose(edges_binary, strel('disk', 2));
%subplot(2,1,2); imshow(edges_binary); title('imclose');
% edges_binary = imfill(edges_binary, 'holes');
% subplot(2,2,3); imshow(edges_binary); title('imfill');
%edges_binary = imopen(edges_binary, strel('disk', 3));
%edges_binary = imopen(edges_binary, strel('line', 3, 0));
% subplot(2,2,4); imshow(edges_binary); title('imopen)');


% Step 6: Filter connected components by roundness & area
cc = bwconncomp(edges_binary);
stats = regionprops(cc, 'FilledArea', 'Area', 'Eccentricity', 'Solidity', 'EulerNumber', 'Perimeter');


% edges_filled = imfill(edges_binary, 'holes');
% cc2 = bwconncomp(edges_filled);
% stats2 = regionprops(cc, 'Area');
% j=1;
% disp(length(stats));
% disp(length(stats2));


final_nodule_mask = false(size(edges_binary));

% Loop through all detected connected components
for k = 1:length(stats)
    % Area filter
    area_ok = stats(k).FilledArea >= Ap2min && stats(k).FilledArea <= 1000;
    
    % Eccentricity and Solidity filters
    round_ok = stats(k).Eccentricity < 0.9 && stats(k).Solidity > 0.3;

    % Compute roundness using FilledArea and Perimeter
    area = stats(k).FilledArea;
    perimeter = stats(k).Perimeter;
    roundness = (4 * pi * area) / (perimeter ^ 2);
    % round2_ok = roundness > 0.75; % Optional

    % Hollowness: Euler number should be 0 or -1 (i.e., not solid object)
    hollowness_ok = stats(k).EulerNumber == 0 | -1;

    % Ratio of original area to filled area (holes check)
    horatio_ok = stats(k).Area / stats(k).FilledArea < 0.9;

    % Combine all conditions
    if area_ok && hollowness_ok && round_ok && horatio_ok % && round2_ok
        final_nodule_mask(cc.PixelIdxList{k}) = true;

        % Print shape statistics
        % fprintf('Nodule %d: Solidity = %.4f, Eccentricity = %.4f, Roundness = %.4f\n', k, stats(k).Solidity, stats(k).Eccentricity, roundness);
    end
end


% Keep only the largest connected component

cc_filtered = bwconncomp(final_nodule_mask);
stats_filtered = regionprops(cc_filtered, 'FilledArea');

if ~isempty(stats_filtered)
    filled_areas = [stats_filtered.FilledArea];
    [~, max_idx] = max(filled_areas);

    % Create new mask with only the largest object
    final_nodule_mask = false(size(edges_binary));
    final_nodule_mask(cc_filtered.PixelIdxList{max_idx}) = true;

    % fprintf('Largest Nodule retained. FilledArea = %.2f\n', filled_areas(max_idx));
end

 
final_filled = imfill(final_nodule_mask,"holes");
final_nodule_mask = final_filled & ~final_nodule_mask;
final_nodule_mask = imdilate(final_nodule_mask, strel('disk', 3));
%final_nodule_mask = imfill(final_nodule_mask,"holes");

figure(100);
subplot(2,2,4); imshow(edges_binary); title('Edges (Post Morphological Cleanup)');
subplot(2,2,3); imshow(final_nodule_mask); title('Final Filtered Nodule Candidates');
hold on;
% Mark the ground truth nodule
if no==0
    plot(voxelX, voxelY, 'ro', 'MarkerSize', 8, 'LineWidth', 2);  % red circle
    text(voxelX + 5, voxelY, 'Nodule GT', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
end
sgtitle('✅ Nodule Detection via Sobel + Morphological Filtering');


gt_grad_mag = grad_mag(round(voxelY), round(voxelX));
fprintf('🔍 Gradient magnitude at GT location: %.4f\n', gt_grad_mag);

% Get properties from the filtered mask (final_nodule_mask)
stats = regionprops(final_nodule_mask, 'Area', 'Eccentricity', 'Solidity');

% Sometimes regionprops returns multiple objects, 
% so find the region that contains GT voxel (round(voxelY), round(voxelX))
gt_label = bwlabel(final_nodule_mask);
gt_region_idx = gt_label(round(voxelY), round(voxelX));
% if gt_region_idx > 0
%     fprintf('➡️ GT Properties (Included):\n');
%     fprintf('Area = %.2f\n', stats(gt_region_idx).Area);
%     fprintf('Eccentricity = %.2f\n', stats(gt_region_idx).Eccentricity);
%     fprintf('Solidity = %.2f\n', stats(gt_region_idx).Solidity);
% else
%     fprintf('➡️ GT not found in any detected region.\n');
% end

% Check if GT voxel is in the initial detected mask
gt_in_mask = final_nodule_mask(round(voxelY), round(voxelX));
fprintf('GT in initial mask: %d\n', gt_in_mask);




BW = final_nodule_mask;  %  detected nodules mask

% 1. Compute distance transform of PROPER mask
D = bwdist(~BW);  % Distance to background (nodules = peaks)

% 2. Find regional maxima (nodule centers)
mask = imextendedmax(D, 2);  % 2 = tolerance for noise

% 3. Modify distance transform to focus on nodules
D2 = imimposemin(-D, ~BW | mask); 

% 4. Apply watershed
L = watershed(D2);

% 5. Create final mask (preserve nodules, remove boundaries)
separated_nodules = BW; 
separated_nodules(L == 0) = 0;  % Remove watershed lines

% 6. Visualize
figure;
imshow(separated_nodules);
title(' Watershed: Separated Nodules');
hold on;
plot(voxelX, voxelY, 'ro', 'MarkerSize', 10); 


% Count nodules before/after watershed
before = bwconncomp(final_nodule_mask).NumObjects;
after = bwconncomp(separated_nodules).NumObjects;

if after > before
    fprintf('✅ Watershed separated %d clusters into %d nodules\n', before, after);
else
    fprintf('⚠️ Check watershed parameters - no separation occurred\n');
end



% 7. Verification
gt_detected = separated_nodules(round(voxelY), round(voxelX));
if gt_detected
    fprintf('✅ Ground truth nodule SUCCESSFULLY detected\n');

    % Calculate physical size and properties from separated nodules mask
    stats = regionprops(separated_nodules, 'Area', 'Eccentricity', 'Solidity');

    area_mm2 = stats(1).Area * (spacing(1) * spacing(2));
    area_mm2 = sqrt(area_mm2*4/pi);
    fprintf('%.2f %.2f\n', ...
            area_mm2, diameter_mm);


else
    fprintf('❌ Ground truth nodule MISSED\n');

end


% 6. Enhanced visualization
figure;
imshowpair(slice_2d, separated_nodules, 'blend');
hold on;
plot(x_circle, y_circle, 'g-', 'LineWidth', 1.5); % Green circle = expected size
plot(voxelX, voxelY, 'r+', 'MarkerSize', 15, 'LineWidth', 2); % Red plus = center
title('Nodule Overlay on Original CT');





