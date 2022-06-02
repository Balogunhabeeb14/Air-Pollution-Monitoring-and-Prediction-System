
The Data used for this project has undergone several preprocessing and thus, if you want a comprehensive data please reach out to me at h.balogun@herts.ac.uk.

Meanwhile, for the columns we have which is similar through out the files. 

There is the datetime/date stamp which represent the date and time the data was collected through the sensor. Mind you the time has been matched with the weather, traffic and the environmental time. 

The zid is the unique ID for the sensors, like i mentioned earlier this is the processed data, we have over 350 sensors in all, but just 14 reflects in this.

The Hour represent the hour of the day in 24 the data was collected

weekdays represent days of the week, 0 represent sunday, 1 represent monday and so on.

Holiday represent wether there is an holiday or not, 0 means no holiday and 1 means there is holiday. this was intuitively proven as we know there is usually a decline in traffic due to holiday.

Humidty, Ambient Pressure, Temperature are weather informations collected around the sensors

Speed represent the average speed of all vehicles around the sensors, this was gotten via Tomtom API.

Green area represent area of the green spaces like the park closest to the sensor.

Road area represent the area of the road closest to the sensor

Buildings represent the avergae heights of buildings closest to the sensors. This was because dispersions of air caused by buildings which practically influence air pollution.

For each file, we have named them after the pollutant of interest, which in most cases represent target, for NO2(1).csv, the NO2 column is expected to be predicted, and same thing applies to other files.

If you still confused, please drop me an email.
H.balogun@herts.ac.uk
