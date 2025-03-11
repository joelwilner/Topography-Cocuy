% Pie Charts master script and RWA
addpath(genpath('/Users/joelwilner/Documents/Dartmouth/Research/Topography')); %Replace with your own path

%% STEP 1: Read in relevant topostats .csv files

% Load topographic statistics from CSV file and convert to table format
topostats = array2table(readmatrix("topostats_refined.csv"));

% Define column names for consistency and readability
colnames = ["ValleySlope", "HeadwallSlope", "CellSize", ...
    "MinElevation", "SlopeTransitionElevation", "HeadwallElevation", ...
    "ValleyWallWidth", "ValleyWallHeight", "Latitude", "Longitude", ...
    "SlopeAspect", "ValleyWidth", "CirqueBinary", "Temperature", ...
    "Precipitation", "GlacierLength", "MaxGlacierThickness", ...
    "GlacierVolume", "PaleoELA", "GlacierArea", "MaxSpeed", "MeanSpeed"];


topostats.Properties.VariableNames = colnames; % Assign the column names to the table


topostats = topostats(topostats.MaxGlacierThickness ~= 0, :); % Remove rows where 'MaxGlacierThickness' is zero

% Load additional dataset if applicable (T5-T8 time period), apply the same formatting
topostats_T5T8 = array2table(readmatrix("topostats_refined_NoPptnGrad_T5-T8.csv"));
topostats_T5T8.Properties.VariableNames = colnames;
topostats_T5T8 = topostats_T5T8(topostats_T5T8.MaxGlacierThickness ~= 0, :);

% Combine the main dataset with the T5-T8 dataset
topostats = vertcat(topostats, topostats_T5T8);

% Load dataset for dry conditions (if applicable) 
topostats_dry = array2table(readmatrix("topostats_refined_NoPptnGrad_dry.csv"));
topostats_dry.Properties.VariableNames = colnames;
topostats_dry = topostats_dry(topostats_dry.MaxGlacierThickness ~= 0, :);

%% STEP 2: RWA and pie chart

%Data Preparation
%Load dataset and define 'data' table for processing
data = topostats;

%Uncomment the following line if you want to filter for a specific precipitation value
%data = topostats_cat(topostats_cat.Precipitation == -50,:);

%Select predictor variables (indices correspond to selected features)
X = data{:, [1, 5, 6, 8, 13, 14, 15]}; %ValleySlope, SlopeTransitionElevation, etc.

X_standardized = zscore(X); %Standardize predictor variables using z-score normalization


Y = data.GlacierLength; %Select the target variable (Glacier Length)

%Linear regression model
mdl = fitlm(X_standardized, Y);
std_coeffs = mdl.Coefficients.Estimate(2:end);


total_variability_explained = sum(std_coeffs.^2); %Compute the total variability explained by all predictors

%Compute the percentage of variability explained by each predictor
percent_variability_explained = (std_coeffs.^2 / total_variability_explained) * 100;

%Display results
disp('Percentage of variability explained by each predictor:');
disp(percent_variability_explained);

%Convert to table for readability
var_table = array2table(percent_variability_explained');

%Define column names corresponding to predictor variables
colnames_vars = ["Valley Slope", "Slope Break Elevation", "Headwall Elevation", ...
    "Valley Wall Height", "Plateau", "Temperature", "Precipitation"];

var_table.Properties.VariableNames = colnames_vars;

%Compute Predictor Weights
%Compute raw weights (squared standardized coefficients)
raw_weights = std_coeffs.^2;
signs = sign(std_coeffs); 
signed_raw_weights = signs .* raw_weights;
rescaled_weights = (raw_weights / sum(raw_weights)) * 100;
signed_rescaled_weights = signs .* rescaled_weights;
weights_matrix = [signed_raw_weights, signed_rescaled_weights];
weights_table = array2table(weights_matrix, 'VariableNames', ...
    {'SignedRawWeights', 'SignedRescaledWeights'}, 'RowNames', colnames_vars);

disp('Weights Table:');
disp(weights_table);

%Variability Explained by Group
%Compute total percentage of variability explained by climatic variables (Temperature, Precipitation)
total_variability_climatic = sum(percent_variability_explained(6:7));

%Compute total percentage of variability explained by topographic variables
total_variability_topographic = sum(percent_variability_explained(1:5));

%Labels for pie chart
group_labels = {['Climatic (' num2str(round(total_variability_climatic)) '%)'], ...
                ['Topographic (' num2str(round(total_variability_topographic)) '%)']};

group_percentages = [total_variability_climatic, total_variability_topographic];

% PIE CHART VISUALIZATION
figure('Position', [100, 100, 1200, 600]);
hold on

% First pie chart: Percentage of variability explained by individual predictors
subplot(1,2,1);
explode = [1 1 1 1 1 0 0]; % Explode effect for specific slices
p = pie(percent_variability_explained, explode, colnames_vars);
set(p(2:2:end), 'FontSize', 22); % Adjust font size for labels

blueColor = [0.004 0.447 0.741]; %Define color schemes
redColor = [0.851 0.325 0.102];

%Define color shades for topographic and climatic predictors
redShades = [1.0 0.6 0.4; 0.95 0.55 0.35; 1.0 0.65 0.45; 0.98 0.6 0.4; 0.96 0.58 0.38];
blueShades = [0.5 0.75 1.0; 0.55 0.8 1.0];


set(p(1), 'FaceColor', redShades(1, :));  
set(p(3), 'FaceColor', redShades(2, :));  
set(p(5), 'FaceColor', redShades(3, :));  
set(p(7), 'FaceColor', redShades(4, :));  
set(p(9), 'FaceColor', redShades(5, :));  
set(p(11), 'FaceColor', blueShades(1, :));  
set(p(13), 'FaceColor', blueShades(2, :));  

%Flip pie chart horizontally by negating XData
for k = 1:2:length(p)
    xData = get(p(k), 'XData');
    set(p(k), 'XData', -xData);
end

%Second pie chart: Variability explained by individual climatic vs. topographic variables
subplot(1,2,2);
p = pie(group_percentages, group_labels);
set(p(3), 'FaceColor', redColor); %Climatic (red)
set(p(1), 'FaceColor', blueColor); %Topographic (blue)
set(p(2:2:end), 'FontSize', 22); 

% Finalize figure
hold off


%% STEP 3: Prepare data for Parameter space subsetting RWA

%Prepare to generate subsets by extracting unique data from each variable
 plateau_bins = unique(data.CirqueBinary)'; 

temp_bins = unique(data.Temperature)'; 
precip_bins = unique(data.Precipitation)'; 
valley_slope_bins = unique(data.ValleySlope)'; 
headwall_elevation_bins = unique(data.HeadwallElevation)'; 
valley_wall_height_bins = unique(data.ValleyWallHeight)';  
slope_break_bins = unique(data.SlopeTransitionElevation)'; 
% Helper function to generate subsets of ranges
generate_subsets = @(bins) arrayfun(@(i) [bins(1), bins(i+1)], 1:length(bins)-1, 'UniformOutput', false);

% Function to generate non-consecutive larger subsets
generate_combined_subsets = @(bins) cellfun(@(comb) [bins(comb(1)), bins(comb(2))], ...
    num2cell(nchoosek(1:length(bins), 2), 2), 'UniformOutput', false);


% Generate all subsets for each parameter
temp_ranges = [generate_combined_subsets(temp_bins)];
precip_ranges = [generate_combined_subsets(precip_bins)];
valley_slope_ranges = [generate_combined_subsets(valley_slope_bins)];
headwall_elevation_ranges = [generate_combined_subsets(headwall_elevation_bins)];
valley_wall_height_ranges = [generate_combined_subsets(valley_wall_height_bins)];
slope_break_ranges = [generate_combined_subsets(slope_break_bins)];
% Plateau binary does not need ranges
plateau_ranges = arrayfun(@(x) x, plateau_bins, 'UniformOutput', false);


% Cartesian product of all possible subset combinations
[temp_grid, precip_grid, slope_grid, headwall_grid, slope_break_grid, valley_wall_height_grid, plateau_grid] = ndgrid(1:length(temp_ranges), 1:length(precip_ranges), ...
    1:length(valley_slope_ranges), 1:length(headwall_elevation_ranges), 1:length(slope_break_ranges), 1:length(valley_wall_height_ranges), 1:length(plateau_ranges));

% Loop through combinations and perform analysis
n_combinations = numel(temp_grid);
%clear results
results = struct('CombinationIndex', repmat({[]}, n_combinations, 1), 'CombinationRanges',repmat({zeros(2,7)}, n_combinations, 1), ...
    'PercentVariability', repmat({zeros(1,7)}, n_combinations, 1), 'TotalVariabilityClimatic', repmat({[]}, n_combinations, 1), 'TotalVariabilityTopographic', repmat({[]}, n_combinations, 1));

warning('off') % Suppress warnings

%%  STEP 4: Parameter space subsetting RWA

start = 23303;

%for idx = 1:n_combinations
for idx = start:n_combinations
    % Extract ranges for the current combination
    temp_range = temp_ranges{temp_grid(idx)};
    precip_range = precip_ranges{precip_grid(idx)};
    slope_range = valley_slope_ranges{slope_grid(idx)};
    headwall_range = headwall_elevation_ranges{headwall_grid(idx)};
    slope_break_range = slope_break_ranges{slope_break_grid(idx)};
    valley_wall_range = valley_wall_height_ranges{valley_wall_height_grid(idx)};
    plateau_value = plateau_ranges{plateau_grid(idx)};

    % Filter data based on these ranges
    subset_data = data(data.Temperature >= temp_range(1) & data.Temperature <= temp_range(2) & ...
                       data.Precipitation >= precip_range(1) & data.Precipitation <= precip_range(2) & ...
                       data.ValleySlope >= slope_range(1) & data.ValleySlope <= slope_range(2) & ...
                       data.HeadwallElevation >= headwall_range(1) & data.HeadwallElevation <= headwall_range(2) & ...
                       data.SlopeTransitionElevation >= slope_break_range(1) & data.SlopeTransitionElevation <= slope_break_range(2) & ...
                       data.ValleyWallHeight >= valley_wall_range(1) & data.ValleyWallHeight <= valley_wall_range(2) & ...
                       data.CirqueBinary == plateau_value, :);
    
    % Perform the regression analysis on the subset of data
    X = subset_data{:, [1,5,6,8, 13,14,15]}; % Using the same variables as before
    X_standardized = zscore(X); % Standardize
    Y = subset_data.GlacierLength; % Target variable
    
    try
        mdl = fitlm(X_standardized, Y); % Fit linear regression model
        std_coeffs = mdl.Coefficients.Estimate(2:end); % Get standardized coefficients 
    catch
        warning('Linear model error; moving to next iteration');
    end
    % Calculate the total and group variability explained
    total_variability_explained = sum(std_coeffs.^2);
    percent_variability_explained = std_coeffs.^2 / total_variability_explained * 100;
    total_variability_climatic = sum(percent_variability_explained(6:7)); % Temp + Precip
    total_variability_topographic = sum(percent_variability_explained(1:5)); % Other variables

    combination_values = [headwall_range',slope_break_range',slope_range',valley_wall_range',repmat(plateau_value,2,1),temp_range',precip_range'];
    
    % Store the results in the struct
    results(idx).CombinationIndex = idx; % Store the index of the combination
    results(idx).CombinationRanges = combination_values; % Store the combination values
    results(idx).PercentVariability = percent_variability_explained'; % Store the percent variability
    results(idx).TotalVariabilityClimatic = total_variability_climatic; % Store total climatic variability
    results(idx).TotalVariabilityTopographic = total_variability_topographic; % Store total topographic variability
    
    disp(idx);
end

%Stopped at 27125


%% STEP 5: Create interval means master

clear intervalMeans
clear intervalMeansMaster
clear intervalMeansMaster_Means
clear meansPerInterval 


% Parameters
numIntervals = 20; % 5% intervals, i.e., 100% / 5%
intervalSize = 5;
numColumns = 7; % Number of variables
xBins = linspace(-100, 100, numIntervals+1); % Bins for X-axis (0 to 100 in 5% steps)

% Variable names for y-axis labels
yAxisLabels = {'Headwall Elevation', 'Slope Break Elevation', 'Valley Slope', ...
               'Valley Wall Height', 'Cirque Binary', 'Temperature', 'Precipitation'};

% Preallocate storage for mean values
meansPerInterval = NaN(numIntervals, numColumns);
%intervalMeansMaster = NaN(47854,numIntervals,numColumns);
intervalMeansMaster = NaN(340200,numIntervals,numColumns); %Joel added 1/6/25

climaticVariability = [results.TotalVariabilityClimatic];
topographicVariability = [results.TotalVariabilityTopographic];

% Define the total number of intervals
%numIntervals = (100 - (-100)) / 10; % Total intervals from -100 to 100 at steps of 5

% Loop over positive integers for interval
for interval = 1:numIntervals
    % Map the positive integer `interval` to the actual lower bound
    lowerBound = -100 + (interval - 1) * 5;
    upperBound = lowerBound + 5;
    
    
    % Subset the data based on TotalVariabilityClimatic for this interval
    %subsetResults = results([results.TotalVariabilityClimatic] >= lowerBound & ...
    %                        [results.TotalVariabilityClimatic] < upperBound, :);
    % Extract the fields into temporary arrays

    
    % Perform the subtraction and apply the logical condition
    subsetResults = results((climaticVariability - topographicVariability) >= lowerBound & ...
                            (climaticVariability - topographicVariability) < upperBound); %Joel added 1/24/25


    %subsetResults = results([results.TotalVariabilityClimatic-results.TotalVariabilityTopographic] >= lowerBound & ...
    %                        [results.TotalVariabilityClimatic-results.TotalVariabilityTopographic] < upperBound, :);
    
    numRows = length(subsetResults); % Get the number of rows in the subset
    if numRows == 0
        continue; % Skip if no data in the interval
    end
    
    % Extract the 2x7 matrices from the 'CombinationRanges' field
    extractedDoubles = cell(numRows, 1); % Create a cell array to store the matrices
    for i = 1:numRows
        extractedDoubles{i} = subsetResults(i).CombinationRanges; % Extract each matrix
    end
    
    % Initialize array to store means for this interval
    intervalMeans = NaN(numRows, numColumns);
    
    % Loop through each cell, extract the mean for each column
    for i = 1:numRows
        currentMatrix = extractedDoubles{i}; % Get the 2x7 matrix
        
        % Calculate the mean of each column in the 2x7 matrix
        intervalMeans(i, :) = mean(currentMatrix, 1);
    end

     for ii = 1:numColumns
         intervalMeansMaster(1:length(intervalMeans),interval,ii) = intervalMeans(:,ii); %Original
    end
    
    % % Store the mean of the interval means for this interval
    % meansPerInterval(interval, :) = mean(intervalMeans, 1);
end

%xBinsRep = repmat(xBins,length(intervalMeansMaster(:,:,6)),1);
% need to ignore zeros

intervalMeansMaster_Means = mean(intervalMeansMaster,1,"omitnan");
% Assuming xBins and intervalMeansMaster_Means(:,:,6) are your x and y data

[optimalXbins1, optimalYbins1] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,1));
[optimalXbins2, optimalYbins2] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,2));
[optimalXbins3, optimalYbins3] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,3));
[optimalXbins4, optimalYbins4] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,4));
[optimalXbins5, optimalYbins5] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,5));
[optimalXbins6, optimalYbins6] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,6));
[optimalXbins7, optimalYbins7] = optimal_bin_size(xBins', intervalMeansMaster_Means(:,:,7));

%% STEP 6: Violin plots

last_non_empty_index = max(find(~cellfun(@isempty,{results.CombinationIndex})));

% Assuming 'results' is your dataset
climaticvalues = cell2mat({results(1:last_non_empty_index ).TotalVariabilityClimatic}');
topographicvalues = cell2mat({results(1:last_non_empty_index ).TotalVariabilityTopographic}');


% Example Influence Data (replace with actual influence data)
topographic_influence = topographicvalues; % Topographic influence
climatic_influence = 100 - topographic_influence;        % Complementary climatic influence

% Calculate Influence Difference
influence_difference = climatic_influence - topographic_influence;

blueColor = [0.004, 0.447, 0.741];
redColor = [0.851, 0.325, 0.102];

figure;
subplot(9,1,[1 2]);
% Split the data
negative_values = influence_difference(influence_difference < 0);
positive_values = influence_difference(influence_difference > 0);

% Plot the negative values (use red/orange color)
hold on;  % To overlay histograms
h1 = histogram(negative_values, 10, 'FaceColor', redColor);  % Red (RGB)
h1.BinWidth = 10;
% Plot the positive values (use blue color)
h2 = histogram(positive_values, 10, 'FaceColor', blueColor);  % Blue (RGB)
h2.BinWidth = 10;
% Adjust the x-axis limits to stretch the plot horizontally
xlim([-max(abs(influence_difference)), max(abs(influence_difference))]);
set(gca, 'XTickLabel', []); % Remove x-axis labels
ylabel('Number of subset combinations')
% Place the count numbers over the bars on the left (for negative values)
for i = 2:length(h1.BinEdges)-7
    count = h1.Values(i);  % Get the count for each bin
    x = h1.BinEdges(i) + (h1.BinEdges(i+1) - h1.BinEdges(i)) / 2;  % Position the label in the center of each bin
    y = count;  % The height of the bar (i.e., the count value)
    text(x, y, num2str(count), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'Color', 'k');
end
set(gca, 'XTick', []); % Remove x-axis labels
legend('Topographic-dominant','Climatic-dominant','Location','Northwest');

hold off

clc

categorynames = ['asdf','asdff'];
numbinsviolin = 20;
C7= repmat([0 0 1], numbinsviolin, 1);
C6 = repmat([0 0.2 1], numbinsviolin, 1);
C5 = repmat([0 0.25 1], numbinsviolin, 1);
C4 = repmat([0 0.6 1], numbinsviolin, 1);
C3 = repmat([0 0.8 1], numbinsviolin, 1);
C2 = repmat([0 1 1], numbinsviolin, 1);
C1 = repmat([0.5 1 1], numbinsviolin, 1);

range_he = 5300-4900;
range_sbe = 4600-4400;
range_vs = 30-5;
range_vwh = 600-200;
range_c = 2;
range_T = 8-1;
range_P = 200;

%intervalEdges = linspace(0, 100, numIntervals+1); % Interval edges: 0, 5, 10, ..., 100
%intervalMidpoints = (intervalEdges(1:end-1) + intervalEdges(2:end)) / 2; % Midpoints: 2.5, 7.5, ..., 97.5

% Create custom tick labels, such as '0-5', '5-10', ..., '95-100'
%tickLabels = arrayfun(@(x, y) sprintf('%d', x), intervalEdges(1:end-1), intervalEdges(2:end), 'UniformOutput', false);



subplot(9,1,3);
violinplot(intervalMeansMaster(:,:,1),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C1,'HalfViolin','left','Bandwidth',range_he*0.1);

ylabel({'Headwall';'elevation (m)'});
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels


   
subplot(9,1,4);
violinplot(intervalMeansMaster(:,:,2),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C2,'HalfViolin','left','Bandwidth',range_sbe*0.1);
ylabel({'Slope break';'elevation (m)'});
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels

subplot(9,1,5);
violinplot(intervalMeansMaster(:,:,3),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C3,'HalfViolin','left','Bandwidth',range_vs*0.1);
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels
    

subplot(9,1,6);
violinplot(intervalMeansMaster(:,:,4),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C4,'HalfViolin','left','Bandwidth',range_vwh*0.1);
ylabel({'Valley wall';'height (m)'});
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels
    

subplot(9,1,7);
violinplot(intervalMeansMaster(:,:,5),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C5,'HalfViolin','left','Bandwidth',range_c*0.1);
ylabel({'Bench';'ternary'});
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels
    

subplot(9,1,8);
ylabel({'Temperature';'deviation (Δ°C)'});
violinplot(intervalMeansMaster(:,:,6),categorynames,'ShowData',false,'ShowMean',true,'ShowMedian',false,...
    'ViolinColor',C6,'HalfViolin','left','Bandwidth',range_T*0.1);

numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

set(gca, 'XTickLabel', []); % Remove x-axis labels
    

subplot(9,1,9);
ylabel({'Precipitation';'deviation (Δ%)'});

% Plot the violinplot
violinplot(intervalMeansMaster(:,:,7), categorynames, 'ShowData', false, 'ShowMean', true, 'ShowMedian', false, ...
    'ViolinColor', C7, 'HalfViolin', 'left', 'Bandwidth', range_P * 0.1);

% Explicitly define ticks and labels
numIntervals = 21;  % Ensure there are 20 intervals
xtick = 1:numIntervals;  % Define tick positions for 20 intervals
xtick_labels = linspace(-100, 100, numIntervals);  % Generate scaled labels from -100 to 100

% Set the tick positions and labels
set(gca, 'XTick', xtick);  % Set ticks at each interval position
set(gca, 'XTickLabel', xtick_labels);  % Scale tick labels to -100 to 100

offset = -0.5;
newXTicks = xtick + offset;
set(gca, 'XTick', newXTicks);

% Adjust the x-axis limits to fully display all violin plots
xlim([0.5, numIntervals-0.5]);  % Add padding around the first and last plot for proper spacing

% Add labels
xlabel('Difference in climatic vs. topographic influence (%): (- topographic-Dominant, + climatic-dominant)', 'FontSize', 14);

%set(gca, 'XTickLabel', []); % Remove x-axis labels








%% Functions

function subsettedTable = subsetTable(table, condition)
    subsettedTable = table;
    for i = 1:size(table, 1)
        if ~eval(condition)
            subsettedTable(i, :) = [];
        end
    end
end

function [optimalXbins, optimalYbins] = optimal_bin_size(x, y)
    % Freedman-Diaconis rule for bin size
    % Bin width = 2 * IQR(x) * length(x)^(-1/3)

    % Calculate IQR for x and y
    IQR_x = iqr(x);
    IQR_y = iqr(y);

    % Get the number of data points
    n_x = length(x);
    n_y = length(y);

    % Apply Freedman-Diaconis rule for bin width
    binWidth_x = 2 * IQR_x * n_x^(-1/3);
    binWidth_y = 2 * IQR_y * n_y^(-1/3);

    % Determine the number of bins for x and y based on the data range
    optimalXbins = ceil(range(x) / binWidth_x);
    optimalYbins = ceil(range(y) / binWidth_y);
end

