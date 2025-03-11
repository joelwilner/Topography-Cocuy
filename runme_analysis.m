
%Replace the following with the paths of choice
workspace_dir = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/workspaces/'; 
statspath = '/Users/joelwilner/Documents/Dartmouth/Research/Topography/stats/';


path = workspace_dir;

% Get a list of files in the directory
fileList = dir([path, '*.mat']);
stats = zeros(length(fileList),13+2+7); %2 climate fields. Extra output fields are: glacier length, max glacier thickness, glacier volume, final ELA, ma and mean speed
mb_master = struct();
counter = 0;


% Iterate over the files
for fileIndex = 1:numel(fileList)
	%try 
		 counter = counter+1; 
		% Construct the full file path
		 filePath = fullfile(path, fileList(fileIndex).name);

		 % Load the file
		 loadedData = load(filePath);
		
         loadedData.actual_numtimesteps = size(loadedData.H_out,3);

		 stats(fileIndex,1) = loadedData.answer(1); %Valley slope
		 stats(fileIndex,2) = loadedData.answer(2); %Headwall slope
		 stats(fileIndex,3) = loadedData.answer(3); %Cell size
		 stats(fileIndex,4) = loadedData.answer(4); %Min elevation
		 stats(fileIndex,5) = loadedData.answer(5); %Slope transition elevation
		 stats(fileIndex,6) =loadedData.answer(6); %Headwall elevation
		 stats(fileIndex,7) =loadedData.answer(7); %Valley wall width
		 stats(fileIndex,8) =loadedData.answer(8); %Valley wall height
		 stats(fileIndex,9) =loadedData.answer(9); %Latitude
		 stats(fileIndex,10) =loadedData.answer(10); %Longitude
		 stats(fileIndex,11) =loadedData.answer(11); %Slope aspect
		 stats(fileIndex,12) =loadedData.answer(12); %Valley width
		 stats(fileIndex,13) =loadedData.answer(13); %Cirque binary

		 stats(fileIndex,14) = loadedData.temp; %temperature departure
		 stats(fileIndex,15) = loadedData.pptn_loop; %precipitation change

		elevation_total_pad = padarray(loadedData.elevation_total,[1 1]);
		elevation_total_pad(elevation_total_pad==0)=-1000;
		padded_rotated = imrotate(elevation_total_pad, loadedData.aspect_deg+loadedData.aspect_correction, 'bilinear', 'loose');
		%padded_derotated = imrotate(padded_rotated, 180-loadedData.aspect_deg-loadedData.aspect_correction, 'bilinear', 'loose'); %Original
		padded_derotated = imrotate(padded_rotated, -loadedData.aspect_deg-loadedData.aspect_correction, 'bilinear', 'loose');
		[firstrow, firstcol] = find(padded_derotated, 1, 'first');
      
		
		for k = 1:loadedData.actual_numtimesteps
            if k>size(loadedData.H_out,3)
                break
            end
			 %Add border around rectangle
			 %if aspect_deg ~= 0 && aspect_deg~= 90 && aspect_deg~=180 && aspect_deg~=270  %If rotated
			 %H_end_derotate = imrotate(loadedData.H_out(:,:,k), 180-loadedData.aspect_deg-loadedData.aspect_correction, 'bilinear', 'loose'); %Original
			 %M_end_derotate = imrotate(loadedData.M_out(:,:,end), 180-loadedData.aspect_deg-loadedData.aspect_correction, 'bilinear', 'loose'); %Original
			 H_end_derotate = imrotate(loadedData.H_out(:,:,k), loadedData.aspect_deg+loadedData.aspect_correction, 'bilinear', 'loose');
          M_end_derotate = imrotate(loadedData.M_out(:,:,k), loadedData.aspect_deg+loadedData.aspect_correction, 'bilinear', 'loose');
			 center_y = round(size(H_end_derotate,1)/2);
			 centerline = fliplr(H_end_derotate(center_y,:)); %IMPORTANT! This was originally within fliplr. 
			 centerline = centerline(loadedData.pad_elev+1:end-loadedData.pad_elev);
			 % else %if not rotated
			 %     H_end_derotate = H_out(:,:,k);
			 %     M_end_derotate = M_out(:,:,end);
			 %     center_y = round(size(H_end_derotate,2)/2);
			 %     centerline = H_end_derotate(center_y,:);
			 % end

			 keyoffset = (length(H_end_derotate)-length(loadedData.H_out))/2;
			 %Start
			 %offset=abs(length(surface_h)-length(centerline));
			 %offsetdiff = abs((length(surface_h)+offset-1)-length(centerline));

			 % disp(['Length of surface_h is ' num2str(length(surface_h))])
			 % disp(['Length of centerline is ' num2str(length(centerline))])
			 % disp(['Length difference is ' num2str(offset)])
			 % error('stop')
			 %trim centerline to same dimensions as surface_h & dist_along_glacier
			 if loadedData.aspect_deg ~= 0 && loadedData.aspect_deg~= 90 && loadedData.aspect_deg~=180 && loadedData.aspect_deg~=270  %If rotated
					centerline = centerline(firstcol+2:length(loadedData.surface_h)+firstcol+2-1);
			 end
			 %centerline = centerline(length(H_end_derotate)-length(elevation_total):end-1);
			 %subplot(2,1,1)
			 %plot(distance_along_glacier/1e3,surface_h+centerline,'Color','blue')
			 %%xlabel('Distance along profile (km)')
			 %ylabel('Elevation (m)')
			 %hold on
			 %plot(distance_along_glacier/1e3,surface_h,'LineWidth',1.5)
			 %hold off
			 %title(['Time = ' num2str(CONFIG.Flow.SaveStep*k) ' yrs']);
			 %pause(0.1)

			 %subplot(2,1,2)
			 %plot(distance_along_glacier/1e3,centerline)
			 %xlabel('Distance along profile (km)')
			 %ylabel('Ice thickness (m)')
			 %ylim([0 max(max(H_out(:,:,end)))+20])
			 %hold on
			 %%title(['Time = ' num2str(CONFIG.Flow.SaveStep*k)]);
			 %pause(0.1)
	%
			 %hold off
			% exportgraphics(gcf,[path 'report/fake_movie/' fullname '_centerline.gif'],'Append',true);
		end

		% Find the indices of the non-zero elements
		%disp(['Size of centerline: ' num2str(size(centerline))]);
		%disp(['Content of centerline: ' num2str(centerline)]);

		nonZeroIndices = find(centerline ~= 0);

		glacierstart = nonZeroIndices(1);
		glacierend = nonZeroIndices(end);

		glacier_length = loadedData.distance_along_glacier(glacierend)/1e3;
		glacier_length_abs = (loadedData.distance_along_glacier(glacierend)-loadedData.distance_along_glacier(glacierstart))/1e3;

		disp(['Glacier length from beginning of domain = ' num2str(glacier_length_abs) ' km. Filename ' filePath])
		%disp(['Absolute glacier length = ' num2str(glacier_length_abs) ' km'])

			dx = loadedData.cs;
			dy = dx;
			vol = sum(sum(loadedData.H_out(:,:,loadedData.actual_numtimesteps)))*dx*dy;
				 %ice area
			  %H=squeeze(loadedData.H_out(trim(1):trim(2),trim(3):trim(4),end));
			  H = loadedData.H_out(:,:,loadedData.actual_numtimesteps);
			  ice_1=double(H>5); %Joel changed this from H>10 to H>5 on 2/6/24
			  ice_area=sum(sum(ice_1))*dx*dy*1e-6; %km^2
			  disp(['Ice Area: ', num2str(ice_area), ' km2'])
			  %ice volume
			  ice_vol=sum(sum(H))*dx*dy*1e-9;
			  disp(['Ice Vol: ', num2str(ice_vol), ' km3'])
			  
			  %ELA
			  newice=loadedData.dem;
			  newice.data=newice.data+H;
			  %M=squeeze(M_out(trim(1):trim(2),trim(3):trim(4),end));
			  M = loadedData.M_out(:,:,loadedData.actual_numtimesteps);
			  U = loadedData.U_out(:,:,loadedData.actual_numtimesteps);	

			  mb=ice_1.*M;
			  %sum(sum(mb))%ALICE - find out why MB sum is +300 m
			  acc1=mean(newice.data(mb>0.5&mb<1));
			  mb0=mb;                         %new mass balance grid
			  mb0(mb>0.5)=-999;               %null where mb is high
			  mb0(mb<-0.5)=-999;              %null where mb is neg
			  
			  mb0(mb==0)=-999;                %null off the ice
			  mb0(newice.data>acc1+100)=-999; %null accumulation area
			  mb0(H<5)=-999;                %null thin ice. I changed this from <50 to <5 on 2/6/24.
			  ELA1=newice.data(mb0>-900);
			  mb0(mb0>-900)=1;
			  mb0(mb0<-900)=0;
			  ELAmat=newice.data.*mb0;     %all ELA values
			  ELA=mean(ELA1);
			  ELAstd=std(ELA1);
				disp(['ELA: ', num2str(ELA), ' m'])
			
			  max_speed = max(max(max(U)));
			  mean_speed = mean(mean(mean(U)));
			  disp(['Max speed: ', num2str(max_speed), ' m/yr']);
			  %disp(['Mean speed: ', num2str(mean_speed), ' m/yr']);

		stats(fileIndex,16) = glacier_length_abs;
		stats(fileIndex,17) = max(max(max(loadedData.H_out)));
		stats(fileIndex,18) = vol; %Glacier volume
		stats(fileIndex,19) = ELA; %ELA
		stats(fileIndex,20) = ice_area; %Glacier area
		stats(fileIndex,21) = max_speed;
		stats(fileIndex,22) = mean_speed; %obsolete

		field = sprintf('Field_%d', fileIndex);	
		mb_master.(field) = M;
	%	stats(fileIndex,21) = AAR; %AAR

		disp(['Current counter is ' num2str(counter)])
		 clear loadedData;
	%catch 
		% if isempty(nonZeroIndices)
		% 	disp(['Current counter is ' num2str(counter) '. Note: glacier length is zero. Filename = ' filePath])
		% end
		% %disp(getReport(e,'extended'));
		% continue
	%end
end

writematrix(stats,[statspath 'topostats_refined.csv']);
save('mb_master_refined.mat', 'mb_master');
%writestruct(mb_master,[statspath 'mb_master_refined_NoPptnGrad_T5-T8_4-16-24.xml']);
