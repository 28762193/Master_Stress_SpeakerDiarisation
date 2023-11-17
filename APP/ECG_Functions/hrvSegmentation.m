function HRV_Frame_Index = hrvSegmentation(hrv_time,wind_size,overlay_size)
%hrvSegmentation: Create 5 minute HRV time frames.

    time_begin = 0;
    add_time = wind_size*overlay_size;
    HRV_Frame_Index = zeros(floor((hrv_time(end)/wind_size)/overlay_size),2);
    time_end = wind_size;
    time_index = 1;
    % Find intervals:
    while time_begin+(wind_size) < hrv_time(end)
        HRV_Frame_Index(time_index,1) = find(ceil(hrv_time)>=time_begin,1,'first');
        HRV_Frame_Index(time_index,2) = find(ceil(hrv_time)>time_begin & floor(hrv_time)<=time_end,1,'last');
        time_end = time_end+add_time;
        time_begin = time_begin+add_time;
        time_index = time_index + 1;
    end
    HRV_Frame_Index(HRV_Frame_Index(:,1)==0,:) = [];

end