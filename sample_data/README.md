## Data Brief

The Data used for this project has undergone several preprocessing steps, so if you want comprehensive data, please reach out to me at h.balogun@herts.ac.uk.

Meanwhile, for the columns we have that are similar throughout the files,

There is the datetime/date stamp which represents the date and time the data was collected through the sensor. Mind you the time has been matched with the weather, traffic and the environmental time. 

The id is the unique ID for the sensors, like I mentioned earlier this is the processed data, we have over 350 sensors in all, but just 14 reflect in this.

The Hour represents the hour of the day in 24 the data was collected

weekdays represent days of the week, 0 represents Sunday, 1 represents Monday and so on.

The holiday represents whether there is a holiday or not, 0 means no holiday and 1 means there is a holiday. this was intuitively proven as we know there is usually a decline in traffic due to holidays.

Humidity, Ambient Pressure, and Temperature are weather information collected around the sensors

Speed represents the average speed of all vehicles around the sensors, this was gotten via Tomtom API.

Green areas represent areas of the green spaces like the park closest to the sensor.

Road area represents the area of the road closest to the sensor

Buildings represent the average heights of buildings closest to the sensors. This was because of dispersions of air caused by buildings which practically influence air pollution.

For each file, we have named them after the pollutant of interest, which in most cases represents a target, for NO2(1).csv, the NO2 column is expected to be predicted, and the same thing applies to other files.

If you are still confused, please drop me an email.
H.balogun@herts.ac.uk
