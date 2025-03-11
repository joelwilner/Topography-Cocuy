%%  Create DEM

clearvars -except numprocs myrank

%Change to your own directory
addpath(genpath('/Users/joelwilner/Documents/Dartmouth/Research/Topography'));

%Change to your own directories of choice
homepath = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/isbrae/jwilner/';
gridpath = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/isbrae/jwilner/grid/';
monthlypath = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/isbrae/jwilner/monthly_data/';
workspace_dir = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/workspaces/';
run_dir = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/runs/';

%Define the parameter values
vs_list = [5 10 15 20 25 30]; %Valley slopes
hs = 30; %Headwall slope
cs = 50; %Cell size
me = 2000; %Min elevation
ste_list = [4400 4500 4600]; %Slope transition elevation
he_list = [4900 5100 5300]; %Headwall elevation
vww = 500; %Valley wall width
vwh_list = [200 400 600]; %Valley wall height
lat = 6.44469; %Cocuy latitude
lon = -72.28585; %Cocuy longitude
a_list = 270; %West-facing
vw_list = 1000;
c = [0 1 2]; %Cirque ternary (0 = just a slope break with no bench, 1 = include the bench, 2 = a special condition for 15° slope the whole way with no headwall and no bench)
kyr = 13; %Late Glacial (which time period do you want to target?)

p_list = [-50 -25 0 25 50]; %Precipitation change (%)
%p_list = [-80 -70 -60]; %Dry precipitation change (%)
t_list = [-1 -2 -3 -4 -5 -6 -7 -8];

answer = zeros(1,13);

%Define a nearly-flat bench
cirque_slope = 0.01;
drop = 0.174533; %m (allows it to be 1 km flat bench.)

% Convert box size from kilometers to degrees
kmPerDegree = 111.32;      % Approximate value for latitude conversion
mPerDegree = kmPerDegree*1000;

%Lapse rate
lps = 0.0058; %lapse rate in -C/m

counter = 0;
LIST = [];

numloops_vs = 6;
%numloops_vs=1; %1/12/25
numloops = 3;
%numloops=1; %1/12/25
numloops_vw = 1;
numloops_a = length(a_list);
%numloops_a = 1; %1/12/25
numloops_pptn = length(p_list);
%numloops_pptn = 1; %1/12/25
numloops_t = length(t_list);
%numloops_t = 1;


for aa=1:numloops_vs %valley slope
    for bb=1:numloops %slope transition elevation
        for cc=1:numloops %headwall elevation
            for dd=1:numloops %Valley wall height
                for ee=1:numloops_a %aspect
                    for ff=1:numloops_vw %valley width
                        for gg=1:numloops_pptn %pptn
                            for hh=1:numloops_t %temperature
                                for iii=1:numloops %bench

											current_filename = ['workspace_fake_T' num2str(t_list(hh)) 'P' num2str(p_list(gg)) 'L58avo_vs' num2str(vs_list(aa)) '_hs30_cs50_me2000_ste' num2str(ste_list(bb)) '_he' num2str(he_list(cc)) '_vww500_vwh' num2str(vwh_list(dd)) '_a270_vw1000_c' num2str(c(iii)) '_Tests_Job'];
											
											prefix = current_filename;
											suffix = '_Iter*.mat';

											% Construct the file path pattern with wildcard for Job and Iter
											filePattern = fullfile(workspace_dir, [prefix, '*', suffix]);

											% Use the dir function to list all files matching the pattern
											files = dir(filePattern);

											% Check if any files match the pattern using exist
											found = false;
											for i = 1:length(files)
											% Check existence of each file
												if exist(fullfile(workspace_dir, files(i).name), 'file')==2
													found = true;
													break
												end
											end

											if found == true
												continue;
											end

											  counter = counter + 1;

                                         % Uncomment the following lines and the associated 'end' if running in parallel
											  %if mod(counter,numprocs)~=myrank
											%	  continue; %Skip, this is done by another MATLAB
											%  else
												   %if exist([workspace_dir current_filename jobnum ],'file')==2
													%	continue;
													%end
													%unique_id = sprintf('Tests_Job%d_Iter%d', myrank, counter);
													%disp(['Processing iteration: ' unique_id]);
                                                    unique_id = num2str(counter); %comment out if running in parallel
													
													LIST = [LIST, counter];
												
													answer(1)=vs_list(aa);
					
													answer(2)=hs;
								
													answer(3)=cs;
										
													answer(4)=me;
													answer(5)=ste_list(bb);
											
													answer(6)=he_list(cc);
									
													answer(7)=vww;
													answer(8)=vwh_list(dd);
										
													answer(9)=lat;
													answer(10)=lon;
													answer(11)=a_list(ee);
													answer(12)=vw_list(ff);
													answer(13)=c(iii);
													answer(13) = 0;
													if answer(13)==2
														 answer(2)=15; %15 degree special slope case
													end


													pptn_loop = p_list(gg);
							
													temp = t_list(hh);
									
		 
													
													if answer(12)<answer(7)
														 error('Valley wall width must be less than the valley width.')
													elseif answer(7)<=answer(3)
														 error('Valley wall width must be greater than the cell size.')
													end

													aspect_correction = -90; %This is very important! It corrects the aspect to be conventional.  added 1/22/24

													valley_slope_deg = answer(1);      % Desired slope angle in degrees
													headwall_slope_deg = answer(2);    % Desired headwall slope angle in degrees
													cellsize = answer(3);              % Size of each cell in meters/cell
													
													min_elevation = answer(4);       % Starting elevation of the valley
													break_elevation = answer(5);      %Slope break transition elevation (m)
													desired_max_elevation = answer(6); % Maximum desired elevation in m
													
											 
													
													% Create valley walls
													wall_width = answer(7);           % Valley wall width in meters
													wall_height = answer(8);          % Valley wall height in meters
													
													% Define the latitude and longitude of Ritacuba Blanco
													centerLat = answer(9);        % Latitude of Ritacuba Blanco
													centerLon = answer(10);      % Longitude of Ritacuba Blanco
													aspect_deg = answer(11);           % Desired slope aspect angle in degrees
													
													width = answer(12);           % Desired valley width (m)
													is_cirque = answer(13);           % Do we have a cirque floor?
		 
													if is_cirque == 1
														 [surface_h, distance_along_glacier] = GlacierBed(desired_max_elevation, min_elevation, width, [], [], [headwall_slope_deg, cirque_slope, valley_slope_deg], [desired_max_elevation, break_elevation, break_elevation-drop, min_elevation], cellsize);
													elseif is_cirque == 0
														 [surface_h, distance_along_glacier] = GlacierBed(desired_max_elevation, min_elevation, width, [], [], [headwall_slope_deg, valley_slope_deg], [desired_max_elevation, break_elevation, min_elevation], cellsize);
													elseif is_cirque == 2 
														 [surface_h, distance_along_glacier] = GlacierBed(desired_max_elevation, min_elevation, width, [], [], [15, 15], [desired_max_elevation, break_elevation, min_elevation], cellsize);
													else
														 error('is_cirque must be 0, 1, or 2')
													end
													
													widthcells = round(width/cellsize); %The valley width (in # of cells)
		 
													elevation_no_walls = repmat(surface_h,widthcells,1);
													
													% Calculate the number of cells for the valley walls based on the cell size
													numCellsValleyWalls = round(wall_width / cellsize);
													
													% Calculate the elevation values for the valley walls
													x_valley_wall = linspace(-wall_width / 2, wall_width / 2, numCellsValleyWalls);
													elevation_valley_wall = (wall_height * (x_valley_wall / (wall_width / 2)).^2);
													
													% Create the elevation matrix for the valley walls
													elevation_valley_wall_matrix = repmat(elevation_valley_wall, size(elevation_no_walls,2), 1)';
		 
													% Add the parabolic valley walls to the elevation matrix
													wall = elevation_valley_wall_matrix(1:numCellsValleyWalls/2,:);
													%wall = cat(1,wall,wall);
													wall_new = zeros(size(wall));
													
													for i=1:size(wall,1) 
														 for j=1:size(wall,2)
															  wall_new(i,j) = wall(i,j)+elevation_no_walls(i,j);
															  if wall_new(i,j)>desired_max_elevation
																	wall_new(i,j)=desired_max_elevation;
															  end
														 end
													end
													
		 
													bottomwall = flipud(wall_new);
													
													% Concatenate the valley walls with the existing elevation values
													elevation_total = cat(1, wall_new, elevation_no_walls, bottomwall);
													%remove middle rows
													midpoint = size(elevation_total,1)/2;
													elevation_total((midpoint-size(wall,1)):(midpoint+size(wall,1)),:)=[];
													
													%If at 90 degree angle, add padding
													% if aspect_deg==0 || aspect_deg==90 || aspect_deg==180 || aspect_deg==270
													pad_elev = 15;
													elevation_total = padarray(elevation_total,[pad_elev pad_elev],min_elevation,'both');
													% else
													%     pad_elev = 15;
													%     elevation_total = padarray(elevation_total,[pad_elev pad_elev],min_elevation,'both');
													% end
													
													originalMask = ones(size(elevation_total));
													% Set the border elements to zero
													originalMask(1:pad_elev, :) = 0;                          % Top border
													originalMask(end-pad_elev+1:end, :) = 0;                  % Bottom border
													originalMask(:, 1:pad_elev) = 0;                          % Left border
													originalMask(:, end-pad_elev+1:end) = 0;                  % Right border
													
													% Rotate the updated elevation matrix by the specified aspect angle
													%rotated_elevation = imrotate(elevation_total, aspect_deg, 'bilinear', 'loose');
													rotated_elevation = imrotate(elevation_total, aspect_deg+aspect_correction, 'nearest', 'loose'); % added +aspect_correction 1/22/24
													rotatedMask = imrotate(originalMask,aspect_deg+aspect_correction,'nearest','loose'); % added +aspect_correction 1/22/24

													rotated_elevation = flipud(rotated_elevation); % added 1/22/24
													rotatedMask = flipud(rotatedMask); % added 1/22/24
													

													rotated_elevation(rotated_elevation == 0) = min_elevation;
													
													%Gaussian Smoothing (optional)
													sigma = 2;
													rotated_elevation = imgaussfilt(rotated_elevation, sigma);
														 
													%Dimensions
													size_m = size(rotated_elevation)*cellsize;
													size_latlon = size_m/mPerDegree; 
													
													% Calculate latitude and longitude limits for the box
													lengthSizeDegrees = (length(surface_h)*cellsize)/ mPerDegree;
													latlim = [centerLat - size_latlon(1)/2, centerLat + size_latlon(1)/2];
													lonlim = [centerLon - size_latlon(2)/2, centerLon + size_latlon(2)/2];
		 
													path = gridpath;
													%path = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/isbrae/jwilner/grid/FakeDEMs_Joel/';
													%save([path 'DEM_valley_parabolic_walls.mat'],'rotated_elevation'); %this is used in isbrae model
													
													name = ['vs' num2str(answer(1)) '_hs' num2str(answer(2)) '_cs' num2str(answer(3)) '_me' num2str(answer(4)) ...
														 '_ste' num2str(answer(5)) '_he' num2str(answer(6)) '_vww' num2str(answer(7)) '_vwh' num2str(answer(8)) ...
														 '_a' num2str(answer(11)) '_vw' num2str(answer(12)) '_c' num2str(answer(13))];
													save([path 'DEM_' name '_' unique_id '.mat'],'rotated_elevation');
													
													%Create a struct with some fields
													elev_info = struct('rotated_elevation',rotated_elevation, ...
														 'centerLat',centerLat, ...
														 'centerLon',centerLon, ...
														 'CellSize',cellsize);
													save([path 'elevinfo_' name '_' unique_id '.mat'],'elev_info');
		 
													%%%%%%%%%%%% Implement Climate %%%%%%%%%%%
		 
													%Monthly Temperature

													path = homepath;
													%path = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/isbrae/jwilner/';
													%load('grid/FakeDEMs_Joel/DEM_valley_parabolic_walls.mat')
													grd = struct(load([path 'grid/DEM_' name '_' unique_id '.mat'])); % added
													load([path 'grid/elevinfo_' name '_' unique_id '.mat']);
													grd.data = grd.rotated_elevation;
													grd.nrows = size(grd.data,1);
													grd.ncols = size(grd.data,2);
													grd.latitude = elev_info.centerLat;
													grd.longitude = elev_info.centerLon;
													grd.xpt = 686321; %change this?
													grd.ypt = 779133; %change this?
													grd.name = 'fake';
													grd.csize = elev_info.CellSize; %needs to change?
													save([path 'grid/fake_dem_' name '_' unique_id '.mat'],'grd');
													%grd.data = zeros(grd.nrows,grd.ncols);
													%save([path 'grid/fake_ice_90_JOEL'],'grd');
													
													
													
													dem = grd;
													
													%Taws = 32.4; %temperature estimated for sea level (C)
													nrows = grd.nrows;
													ncols = grd.ncols;
													
													secday = 24*60*60;
													mo_num = [31 28 31 30 31 30 31 31 30 31 30 31];
													
													% Monthly Precipitation
													%annual mean precip is from Ceballos et al. 2006
													pptn = 1100*ones(nrows,ncols);
													grd.data = pptn;
													save([monthlypath 'annualmeanpptnsurface_' name '_' unique_id '.mat'],'grd')
													%monthly proportions of precip caluclated from VDH etal 1981
													%prop = [0.02 0.01 0.05 0.12 0.13 0.11 0.08 0.07 0.09 0.14 0.12 0.06];
												   prop = [0.03 0.03 0.06 0.13 0.12 0.09 0.07 0.06 0.08 0.14 0.12 0.06]; %From colo_mocli.m

													% Temperature in C from Ceballos etal 2010
													%Ta_mo = [2.4 2.1 2.2 2.6 2.9 2.7 2 2.4 2.5 2.3 2.7 2.3]; %Original
													Ta_mo = [2.35 2.15 2.2 2.6 3.13 2.95 1.98 2.38 2.52 2.35 2.73 2.4]; % added 2/2/24, based on guide Google doc
													
													%Nevado del cocuy (Ceballos etal 2010, pg 78)
													Zaws = 4150; %Elevation of meteorological station (m asl)

													% Relative humidity in fraction between 0 and 1
													% rh = 60:0.2:90;%(VDH 1981)
													% rh(150:ncols) = 90;
													% RH_mo = ones(nrows,ncols);
													% for nn = 1:nrows
													%     RH_mo(nn,:) = rh/100;
													% end
													RH_mo = 0.90; %ceballos et al 2011 *average the 80-100 range pg.76
													
													% wind speed in m/s
													U_mo = 2.5;
													% got wind info from article about el cocuy (2008-09) ceballos et al 2010
													% old:Weather Spark website for Cocuy Colombia= 2.2
													
													%solar radiation in W/m2 needs to be converted to MJ/m2
													Im_moW = [700 1080 865 650 633 615 580 490 620 500 600 780];
													%ceballos et al 2008 at pg64
													%OLD:200 W/m2 from Wangon etal 1999 Zongo Glacier in Bolivia
													%converting W/m2 to MJ/m2
													Im_mo = Im_moW*24*60*60/1e6;
													
													clear Im_month U_month RH_month Ta_month P_month
													
													for month = 1:12
														 %Solar radiation in MJ/m2
														 Im_month(month,:,:) = Im_mo(month)*ones(nrows,ncols);
														 %wind speed in m/s
														 U_month(month,:,:) = U_mo*ones(nrows,ncols);
														 %relative humidity 0 to 1 (0.6=60%) fraction
														 RH_month(month,:,:) = RH_mo; %RH_mo*ones(nrows,ncols);
														 %Mean monthly temperature deg C
														 Ta_month(month,:,:) = Ta_mo(month)-lps*(dem.data-Zaws);
														 %Total monthly precipitation (mm/a to mm/month converted to mm/sec)
														 P_month(month,:,:) = pptn*prop(month)/(mo_num(month)*secday);
													end
													
													save([monthlypath 'monthlymeanclimate58_' name '_' unique_id '.mat'],'P_month','Ta_month','U_month','Im_month','RH_month')
													
													% added the following to produce the ice file
													path = homepath;
													
													grd.data = zeros(grd.nrows,grd.ncols);
													save([path 'grid/fake_ice_' name '_' unique_id '.mat'],'grd');
													
		 
													%%%%%%%%%%%%%% Flow Model %%%%%%%%%%%%%%%%
													
													%save fake_fl CONFIG
													path = homepath;
													
													%CONFIG setup
													global CONFIG
													
													addpath(genpath('grid'))
													addpath(genpath('matdata'))
													addpath(genpath('mfiles'))
													addpath(genpath('monthly_data'))
													addpath(genpath('runs'))
													%load mfiles/config/colo_flow
													RUN='tempsense';'lpssense';'trans';'std';'paramfit';'fake_setup';
													%'whitenoise';'noavo';'presentday';'ebmtest';'std';'us';'flow_sense';'sense';
													%MORA={'Afr11';'Moul17';'Maho20';'Maho23'};%
													MR=1;%model run number
													
													CONFIG.Grid.TerrainFile = [path 'grid/fake_dem_' name '_' unique_id];
													CONFIG.Grid.IceFile = [path 'grid/fake_ice_' name '_' unique_id];
													%CONFIG.Grid.Bedfile = 'grid/bedgrid.mat';
													CONFIG.Title = 'Glaciers'; %full title
													CONFIG.Name = 'fake'; %short, no caps, no spaces
													%CONFIG.Location.longitude = -72; % commented out after model runs 11/26/23
													CONFIG.Location.longitude = lon;
													%CONFIG.Location.latitude = 0; % commented out after model runs 11/26/23
													%CONFIG.Location.latitude = 6.44469; 
													CONFIG.Location.latitude = lat; %Test latitude;  added 12/29/23
													CONFIG.Location.UTC = -5;
													CONFIG.Info.BasePath = path;
													CONFIG.Info.ClimatePath = [path 'grid'];
													CONFIG.Info.RunPath = [path 'matdata'];
													CONFIG.Info.TempPath = [path 'matdata'];
													CONFIG.Terminus.ReferencePoint = [];
													%CONFIG.TempPath = 'BigTemp';
													
													%=================Energy Balance Model
													EB.RunName = 'fake_90_';
													EB.KyrBP = kyr; 
													%EB.KyrBP = 0;%Make sure this relates to the moraine. Joel commented out 11/26/23
													EB.dt = 1;%WHERE IS THIS USED?
													%EB.TempLapseRate = -0.0065; %Need to change mocligrids if you change this
													EB.TempLapseRate = -lps;%Changed 11/11/23
													EB.TempChange = 0;
													EB.PptnChange = 0;
													EB.WindSpeedChange = 0;
													EB.RHChange = 0;
													EB.SolarChange = 0;
													EB.SnowTempThreshold = 2.5; %Molg et al., 2008
													EB.MonthlyData = [path 'monthly_data/monthlymeanclimate58_' name '_' unique_id '.mat'];%lps6.5
													EB.TempSource = '';
													EB.PptnSource = '';
													EB.WindSource = 'reanalysis';
													EB.RHSource = 'reanalysis';
													EB.InputDataGridDir = [path 'monthly_data'];
													%=======Albedo
													EB.AlbedoMode = 'ela';
													EB.ELA = 4900; %From Google Earth imagery
													EB.dc = 0.11;%characteristic scale for snow depth
													EB.tc = 21.9;%time scale how fast snow albedo->firn albedo
													EB.Zice = 0.004;%roughness of ice
													EB.Zsnow = 0.001;%roughness of snow
													EB.Zthick = 0.5;
													EB.Zheight = 2;
													EB.SnowAlbedo = 0.75;
													EB.FirnAlbedo = 0.53;
													EB.IceAlbedo = 0.34;
													
													EB.SaveGrid = 0;
													EB.PptnBaseElev = NaN;
													EB.PptnFunction = 'reanalysis';
													EB.PptnFactor = 1;
													EB.PptnSurface = [path 'monthly_data/annualmeanpptnsurface_' name '_' unique_id];
													EB.SnowAccumFun = 'linear';
													EB.SnowDepthInit = 0;
													EB.SnowEventThreshold = 0.005;
													EB.SnowMaxThickness = 20;
													EB.SnowLineElevation = 4900;
													EB.SnowGradient = 0.002;
													EB.SnowThicknessReset = 0;
													EB.DebrisCover = 0;
													EB.DebrisMeltFactor = 0.1;
													EB.CalcHydrology = 0;
													EB.PointsOfInterest = [];
													EB.TimesOfInterest = [];
													EB.SnowlineTimes = [];
													EB.RunSuffix = '';
													EB.InputDataFun = '';
													EB.InputDataFunMode = 'single';
													EB.StartDate = [];
													EB.EndDate = [];
													EB.SeasonalSnowFactor = 1;
													EB.InsoTimeStep = 0;0.0417;%WHERE IS THIS USED?
													%=======Day to day variability
													%EB.Sigma = 0;1;2.56;%daily Temp variability within a month (NZ value 2.56C) %Joel commented out 11/26/23
													EB.Sigma = 1; %Joel revised this 11/26/23
													EB.SigmaPptn=0;%pptn CHECK THIS
													
													%=======SnowSlide Model (MTD)
													EB.MGTThreshold = 1;%0=no avo model %Joel changed from 0.02 to 1 on 11/26/23 after model runs
													EB.SnowDispr = 0.1;
													EB.MinSlope = 45;
													EB.SnowThicknessLimit = 0.2;
													EB.gamma = 0.125; 
													CONFIG.EnergyBalance = EB;
													
													
													CONFIG.Flow.SaveStep = 10;
													CONFIG.Flow.SaveFile = [path 'runs/fakecheck'];%CHECK
													CONFIG.Flow.PlotStep = 1000;
													CONFIG.Flow.PlotVariable = {'mass_balance','ice_thickness'};
													%{'sliding_velocity','deformation_velocity','ice_velocity','ice_thickness','mass_balance'};
													CONFIG.Flow.PlotMinThickness = 10;%ice must be >10 m thick
													CONFIG.Flow.CAxis = [-15,5;0,500];%[0,20;0,30;0,60;0,300;-10,10];
													CONFIG.Flow.PlotMovie = 0;
													%=================2D Flow Model
													CONFIG.Flow.RunName = 'fake';
													CONFIG.Flow.CameraPosition = [];
													CONFIG.Flow.CameraViewAngle = [];
													CONFIG.Flow.RestartFile = '';'runs/fake_restart'; %MAKE
													CONFIG.Flow.RestartIndex = -1;%look this up %Commented out 11/26/23
													%CONFIG.Flow.RestartIndex = 100; %Changed to this 11/26/23
													CONFIG.Flow.StartTime = 0;
													CONFIG.Flow.StopCond = 'FixedTime';
													%CONFIG.Flow.StopCond = 'NoChange'; %Joel added this 11/3/23. Doesn't work.
													CONFIG.Flow.StopNum = 500; %#of model years
													CONFIG.Flow.TimeStep = 0;%variable
													%CONFIG.Flow = rmfield(CONFIG.Flow,'ActualStopNum'); %Joel added as a test
													
													
														 %variable US with elevation
														 %load(CONFIG.Grid.TerrainFile)
														 %USChar=ones(grd.nrows,grd.ncols);
														 %USChar(grd.data<4000)=50; 
														 %USChar(grd.data>4000)=20;
													
													CONFIG.Flow.UsChar = 20; %Kessler et al., 2006 ~20m/a
													CONFIG.Flow.TaubChar = 100000;
													
													CONFIG.Flow.FlowDeform = 1e-17;%Pa-3a-1LemeurandVincent;
													CONFIG.Flow.FlowSlide = 4.5e-20;
													CONFIG.Flow.TopoMeanBlockSize = 1;
													CONFIG.Flow.MaxSpeed = Inf;
													CONFIG.Flow.MaxTimeStep = 0.1;
													CONFIG.Flow.MassBalanceFunction = 'M=flow_ebm_monthly_daily(grd_noice,H,t);';
													%'M=0;for st=1:5;clear flow_ebm_monthly_daily;M=M+flow_ebm_monthly_daily(grd_noice,H,t)/5;end;';
													CONFIG.Flow.MassBalanceStep = 10;% how often to call mass balance model
													CONFIG.Flow.Calving = 0;
													CONFIG.Flow.NoIceDist = 4000;%used in 'fit' mode?
													
													%%%%%%%%%
													%Create mask:
                                                    path = homepath;
													load([path 'grid/fake_dem_' name '_' unique_id '.mat']);
													%grd.data = double(grd.data>min(min(grd.data)));
													grd.mask = rotatedMask; %Changed from grd.data to grd.mask 11/23/23
													save([path 'grid/fake_mask_' name '_' unique_id '.mat'],'grd');
													CONFIG.Flow.CalcMask = [path 'grid/fake_mask_' name '_' unique_id '.mat'];
													% grd.data = double(grd.data>=0);
													% save([path 'grid/fake_mask_90_JOEL'],'grd');
													% CONFIG.Flow.CalcMask = [path 'grid/fake_mask_90_JOEL'];
													%%%%%%%%
													
													
													%==================Climate for Brian
													CONFIG.Climate.InterpUpdateTemp = 1;
													CONFIG.Climate.InterpOnlyTemp = 0;
													CONFIG.Climate.InterpFrostCycles = 0;
													CONFIG.Climate.InterpYears = 0;
													
													
													
													save([path, 'mfiles/config/fake_lgmis_' name]) %for isbrae
													  
													clear flow_ebm_monthly_daily
													% load bed elevation
													
													switch RUN      
														 case 'tempsense'
													
															  pptn = pptn_loop;
															  temp = temp;
															  %temp = [-8 -10 -12];
															  CONFIG.EnergyBalance.TempLapseRate = -lps;
															  CONFIG.EnergyBalance.MonthlyData = [path 'monthly_data/monthlymeanclimate58_' name '_' unique_id];
															  %lps = CONFIG.EnergyBalance.TempLapseRate*-10000; %make sure this matches what is above
															  CONFIG.Flow.StopNum = 30; %Suggestion: 250 for small runs, 400 for long runs
															  CONFIG.Flow.RestartFile = '';'runs/sensitivity/colo_T-68P-50L58unip.mat';
															  for mm=1:length(temp)
																	CONFIG.EnergyBalance.TempChange = temp(mm);
																	CONFIG.EnergyBalance.PptnChange = pptn(mm);
																	CONFIG.EnergyBalance.KyrBP=kyr;%Changed from 20 to 30 on 11/26/23
																	CONFIG.Flow.PlotStep = 4000;
																	fullname = ['fake_T',...
																		 num2str(CONFIG.EnergyBalance.TempChange*10),'P',...
																		 num2str(pptn(mm)),'L58avo_',name];
																	CONFIG.Flow.SaveFile=[run_dir fullname '_' unique_id];
																	save([path, 'mfiles/config/fake_lgme_' name])
																	
                                                                    [H_out,t_out,U_out,M_out]=flow_2d; %Contact Alice Doughty for flow_2d function and other necessary functions
															        ice_binary=double(H_out(:,:,1)>10);
																	save([workspace_dir '/workspace_' fullname '_' unique_id])
                                                                   
															  end
															  
															 
															  
														 case 'lpssense'
															  pptn=[-10 -10 -30 -30 -30 -40 -40 -40];
															  temp=[-3.2 -3.3 -3.9 -4 -4.1 -4.4 -4.5 -4.6];
															  CONFIG.EnergyBalance.MonthlyData = [path 'monthly_data/colo_dem_30_monthlymeanclimate67'];
															  CONFIG.EnergyBalance.TempLapseRate = -0.0067;
															  lps = CONFIG.EnergyBalance.TempLapseRate*-10000; %make sure this matches what is above
															  CONFIG.Flow.StopNum = 350; %250 for small runs, 400 for long runs
															  for mm=1:length(temp)
																	CONFIG.EnergyBalance.TempChange = temp(mm);
																	CONFIG.EnergyBalance.PptnChange = pptn(mm);
																	CONFIG.EnergyBalance.KyrBP=21;
																	CONFIG.Flow.PlotStep = 1000;
																	CONFIG.Flow.SaveFile=['runs/sensitivity/colo_T',...
																		 num2str(CONFIG.EnergyBalance.TempChange*10),'P',num2str(pptn(mm)),'L',num2str(lps),'avo'];
																	save mfiles/config/colo_lgmlps
																	[H_out,t_out,U_out,M_out]=flow_2d;
															  end
															  
														 case 'trans'
															  disp('trans')
															  load paleo/RutunduT %html/MahomaT
															  CONFIG.EnergyBalance.TempChange=fliplr(RutunduT.temp);%MahomaT.temp;
															  CONFIG.EnergyBalance.KyrBP=fliplr(RutunduT.date);%MahomaT.date;
															  CONFIG.Flow.PlotStep = 50;
															  CONFIG.Flow.SaveStep = 50;
															  CONFIG.Flow.MassBalanceStep = 50;% how often to call mass balance model
															  CONFIG.Flow.SaveFile = 'runs/transrut';%CHECK
															  CONFIG.Flow.PlotVariable = {'ice_thickness'};
															  %{'sliding_velocity','deformation_velocity','ice_velocity','ice_thickness','mass_balance'};
															  CONFIG.Flow.CAxis = [0,500];
															  CONFIG.Flow.PlotMovie = 1;
															  CONFIG.Flow.StartTime = 0;
															  CONFIG.Flow.StopNum = 14000;
															  save mfiles/config/colo_lgmis
															  [H_out,t_out,U_out,M_out]=flow_2d;
															  
														 case 'fake_FL'
													hwheight = 5100; %headwall height (5100, 5200)
													foreci = 4550; %forecirque (4550, 4200) base of slope below cirque;
													valley = 3600; %valley mid elevation (3800, 3600)
													foremor = 3200; %(3350, 3200)
													xx=4;
																	elev = [hwheight 4600 4550 foreci 4000 valley foremor];
																	dist = [0 1980 2610 3780 5670 8820 10710];
																	fc.xStep = 90;
																	fc.ifli{1,xx} = 0:90:10710;%xStep = 90
																	fc.b{1,xx} = interp1(dist,elev,fc.ifli{1,xx}); %bed elevation
																	fc.h{1,xx} = fc.b{1,xx}; %bed elev is the same as ice height
																	fc.w{1,xx} = 500*ones(size(fc.b{1,xx})); %valley width 500 m
																	fc.s{1,xx} = (fc.b{1,xx}(1,1:end-1) - fc.b{1,xx}(1,2:end))/fc.xStep;
																	fc.s{1,xx}(1,119) = 0;%hardcoded for now so lines are the same length
																	plot(fc.ifli{1,xx},fc.b{1,xx})
																	fc.flx{1,xx} = 699500*ones(length(fc.b{1,xx}))-fc.ifli{1,xx};
																	fc.fly{1,xx} = 785780*ones(length(fc.b{1,xx}));
													
														 case 'colo_fake'
															  load grid/colo_dem_90
															  dem = grd.data(100:130,30:152);
															  dem = fliplr(dem);
															  dem = round(dem,-1);
															  dem(dem>5200) = 5200; %cut off peaks
															  dem(dem<3200) = 3200; %fill in lower valley
															  dem(dem(:,1:31)<4550) = 4550;
															  dem(dem(:,1:45)<4500) = 4500;
															  grd.data = [];
															  grd.data(1:31,1:123) = dem;
															  grd.data(32:62,1:123) = dem;%hwh lower
															  grd.data(63:93,1:123) = dem;%cirq lower
															  grd.data(94:124,1:123) = dem;%valley higher
															  hw = dem(:,1:25); %4600 at [25 17] hwh=5200
															  hw(hw>5100) = 5100; %hwh=5100
															  grd.data(32:62,1:25) = hw;
															  
															  fcirq = dem(:,26:64); %4000 at [64 16], cirq = 4550 at [31 14], 4500 at []
															  elev = [4550 4200 4000];
															  dist = [6 20 39];
															  elev2 = interp1(dist, elev, 1:39);
															  elev2 = round(elev2,-1);
															  fcirq(15,6:end) = elev2(6:end);
															  fcirq(16,6:end) = elev2(6:end);%set center of valley floor first
															  diff = (fcirq(11,8:36) - fcirq(15,8:36))/4;
															  fcirq(14,8:36) = fcirq(15,8:36)+diff;
															  fcirq(13,8:36) = fcirq(14,8:36)+diff;
															  fcirq(12,8:36) = fcirq(13,8:36)+diff;
															  diff = (fcirq(20,8:36) - fcirq(16,8:36))/4;
															  fcirq(17,8:36) = fcirq(16,8:36)+diff;
															  fcirq(18,8:36) = fcirq(17,8:36)+diff;
															  fcirq(19,8:36) = fcirq(18,8:36)+diff;
															  fcirq(17:18,37:38) = fcirq(17:18,35:36);
															  fcirq = round(fcirq,-1);
															  grd.data(63:93,26:64) = fcirq;
															  
															  valley = dem(:,65:end); %valley = 3500 at [101 23] to 3200
															  elevb = [4000 3800 3350];
															  dist = [64-64 101-64 123-64];
															  elev3 = interp1(dist, elevb, 1:59);
															  for ii=1:59
															  find(valley(:,ii)<elev3(ii));
															  valley(ans,ii) = elev3(ii);
															  end
															  grd.data(94:124,65:end) = valley;
															  grd.data = fliplr(grd.data);
															  
															  grd.ncols = 123;
															  grd.nrows = 124;
															  grd.latitude = 0;
															  grd.longitude = -72;
															  grd.name = 'fake';
															  
													%         save fake_dem_90 grd
													%         grd.data = zeros(124,123);
													%         save fake_ice_90 grd
															  
														 case 'fake_setup'
															  %Flowline setup
													hwheight = [5100 5200 5400]; %headwall height (4700-5400)
													cifl = [4200 4300 4400 4500 4600]; %cirque floor (4300-4600)
													foreci = 4000; %forecirque (base of slope below cirque)
													valley = [3500 3600 3700 3800 3900]; %valley end elevation
													foremor = 3400; %foremoraine, elevation beyond moraines
													xx = 0;
													for nn = 1:5
														 for mm = 1:5
															  for pp = 1:5
																	xx = xx+1;
																	elev = [hwheight(nn) cifl(mm) cifl(mm) foreci valley(pp) valley(pp)-100 foremor];
																	dist = [0 1500 2200 4500 7500 10000 13200];
																	fc.xStep = 90;
																	fc.ifli{1,xx} = 0:90:13200;%xStep = 90
																	fc.b{1,xx} = interp1(dist,elev,fc.ifli{1,xx}); %bed elevation
																	fc.h{1,xx} = fc.b{1,xx}; %bed elev is the same as ice height
																	fc.w{1,xx} = 500*ones(size(fc.b{1,xx})); %valley width 500 m
																	fc.s{1,xx} = (fc.b{1,xx}(1,1:end-1) - fc.b{1,xx}(1,2:end))/fc.xStep;
																	fc.s{1,xx}(1,147) = 0;%hardcoded for now so lines are the same length
																	plot(fc.ifli{1,xx},fc.b{1,xx})
																	fc.flx{1,xx} = 699500*ones(length(fc.b{1,xx}))-fc.ifli{1,xx};
																	fc.fly{1,xx} = 785780*ones(length(fc.b{1,xx}));
																	hold on
															  end
														 end
													end
													fc.PlotOn = 0;
													fc.g = 9.81;
													fc.rho = 910;
													fc.taub = 150000;
													fc.MaxDepth = 250;
													fc.HeadBuff = 8;
													fc.TermBuff = 10;
													fc.CutBuff = 4;
													fc.MaxSearchSpeed = 4;
													fc.IgnoreRocks = 1;
													fc.Smoothing = 3;
													fc.TimeStepType = 'CFL';
													fc.TimeStepNum = 0;
													fc.MassFun = 'colo_mb';%Need a refresher on what this looks like, monthly gridded climate?
													fc.MaxTime = 100;
													fc.PlotInt = 1;
													fc.FdefConst = 1.5e-24;
													fc.FsldConst = 4.5e-20;
													fc.SolnMethod = 'Explicit';
													fc.InitialCond = 'Grid';
													fc.StopCond = 'NoChange';
													fc.StopNum = 1e-4;
													fc.BedSource = 'res';
													fc.SolnOmega = 0;
													fc.StartTime = 0;
													
													fc.LineName = {'RioNegro'};
													fc.TimeStepType = 'CFL';
													fc.TimeStepNum = 0;
													fc.UpperBCType = {'FixedHeight'};
													fc.UpperBCNum = {[0]};
													fc.LowerBCType = {'FixedHeight'};
													fc.LowerBCNum = {[0]};
													fc.CloseGrid = {[]};
													fc.xCode = {''};
													
													
													
													CONFIG.Flowline = fc;
															  
													end            
    
												%end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

disp(LIST)
