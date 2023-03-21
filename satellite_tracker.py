from random import choice, uniform
import os
import urllib.request
import numpy as np

from skyfield import api
from skyfield.api import load

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Communication Satellites-Geostationary orbit
comm_sats_geostat = {	'INSAT-4B': 30793, 'INSAT-4CR': 32050, 
						'GSAT-6': 40880, 'GSAT-7': 39234, 
						'GSAT-8': 37605, 'GSAT-9': 42695, 
						'GSAT-10': 38779, 'GSAT-12': 37746, 
						'GSAT-14': 39498, 'GSAT-15': 41028, 
						'GSAT-16': 40332, 'GSAT-17': 42815, 
						'GSAT-18': 41793, 'GSAT-19': 42747, 
						'GSAT-29': 43698, 'GSAT-11': 43824, 
						'GSAT-7A': 43864, 'GSAT-31': 44035, 
						'GSAT-30': 45026}
# Earth Observation-Sun-synchronous orbit
eart_sats_sunsync = {	'RESOURCESAT-1': 28051, 'RESOURCESAT-2': 37387, 
						'RESOURCESAT-2A': 41877, 'CARTOSAT-1': 28649, 
						'CARTOSAT-2': 29710, 'CARTOSAT-2A': 32783, 
						'CARTOSAT-2B': 36795, 'CARTOSAT-2C': 41599, 
						'CARTOSAT-2D': 41948, 'CARTOSAT-2E': 42767, 
						'CARTOSAT-2F': 43111, 'CARTOSAT-3': 44804, 
						'RISAT-1': 38248, 
						'RISAT-2B': 44233, 'RISAT-2BR1': 44857, 
						'OCEANSAT-2': 35931,
						'SARAL': 39086, 'SCATSAT 1': 41790, 
						'HYSIS':43719, 'RISAT-2BR2':46905}
# Earth Observation-Geostationary orbit
eart_sats_geostat = {	'INSAT-3D': 39216, 'INSAT-3DR': 41752}
# Regional Navigation-Geostationary orbit
regi_navi_geostat = {	'IRNSS-1A': 39199, 'IRNSS-1B': 39635, 
						'IRNSS-1C': 40269, 'IRNSS-1D': 40547, 
						'IRNSS-1E': 41241, 'IRNSS-1F': 41384, 
						'IRNSS-1G': 41469, 'IRNSS-1I': 43286}
# Scientific 
scie_sats = {			'ASTROSAT': 40930}
# Experimental
expr_sats = {			'INS-1A': 41949, 'INS-1B': 41954, 
						'INS-1C': 43116, 'PRATHAM':41783 }
# Not available
# scie_sats_2 =  {'MARS ORBITER MISSION': 39370, 'CHANDRAYAAN 2': 44441}
	
# Returns a dictionary of {satellites:IDs} operated by ISRO
def getISROSatelliteList():
	dict = {}
	dict.update(comm_sats_geostat)
	dict.update(eart_sats_sunsync)
	dict.update(eart_sats_geostat)
	dict.update(regi_navi_geostat)
	dict.update(scie_sats)
	dict.update(expr_sats)
	return dict

# Returns a dictionary of {satellites:IDs} operated by ISRO
def getISROGEOSatelliteList():
	dict = {}
	dict.update(comm_sats_geostat)
	dict.update(eart_sats_geostat)
	return dict
	
# Returns a dictionary of {satellites:IDs} operated by ISRO
def getISRONavSatelliteList():	
	dict = {}
	dict.update(regi_navi_geostat)	
	return dict
		
# Returns a dictionary of {satellites:IDs} operated by ISRO
def getISROLEOSatelliteList():	
	dict = {}	
	dict.update(eart_sats_sunsync)
	dict.update(scie_sats)
	dict.update(expr_sats)
	return dict

# delete an entry in any of the groupings, if exists.
def deleteDictEntry(satname):
	if satname:
		comm_sats_geostat.pop(satname, None)
		eart_sats_sunsync.pop(satname, None)
		eart_sats_geostat.pop(satname, None)
		regi_navi_geostat.pop(satname, None)
		scie_sats.pop(satname, None)
		expr_sats.pop(satname, None)
	return

# Saves a TLE file (filename) locally for the group of satellites 
# provided in the dictionary
def saveTLE(dictionary, filename):
	file_out = open(filename, "w")	# create a new file 
	base = 'http://celestrak.com/cgi-bin/TLE.pl?CATNR='
	for satname, satid in dictionary.items():
		url = base + str(satid)	
		with urllib.request.urlopen(url) as fd:
			with open(filename, 'rb+') as file:
				file.seek(len(file.read())) # seek the end of file 
				b = bytearray(fd.read())	# read as `bytes` object
				tmp = str(b, 'utf-8') #...and now we convert it into string				
				if not tmp.startswith('No TLE found'):
					file.write(b)				# and append-add it
				else:
					print('TLE not found for ' + satname + '[' + str(satid) + ']' )					
					deleteDictEntry(satname)					
	file_out.close() 						# close the handle
	return		

# Returns (sats_data) a dictionary of skyfield EarthSatellite object with name and ids
# as individual elements {name:sat, id:sat, ....}
def readTLE(fileName='./india_tle.dat'):
	# tle referesh days
	refresh_days = 14
	# current date and time
	ts = load.timescale(builtin=True)
	t  = ts.now()

	days = 1; old_date = 0;
	if not os.path.exists(fileName):
		# call create local tle function
		print("Creating new tle file")
		dict = getISROSatelliteList()
		saveTLE(dict, fileName)		
		# load the new one
		satellites = load.tle(fileName)
	else:
		# load the local file		
		satellites = load.tle(fileName)
		# get the first in the dictionary
		sat_id = list(satellites.keys())[0] 
		satellite = satellites[sat_id]

		days = t - satellite.epoch
		old_date = satellite.epoch.utc_strftime('-%Y-%m-%d-%H-%M-%S')		
	# if older than refresh_days create new local tle and load new one 
	if abs(days) > refresh_days:
		# call create local tle function
		print("Creating new tle file")
		# backup old tle
		backupFilename = Path(fileName).stem + old_date + '.dat'		
		os.rename(fileName, backupFilename)
		# get the new one
		dict = getISROSatelliteList()
		saveTLE(dict, fileName)		
		# load the new one
		satellites = load.tle(fileName)
	# sats dictionary for easy search
	sats = {}
	for item in [satellites]:
		names = [key for key in item.keys()]
		for satname in names:
			sat 			= item[satname]
			satid 			= sat.model.satnum
			sats[satid] 	= sat
			sats[satname]   = sat		
	# return dictionary
	return sats
  
def getSatNameFromId(satid):
	satname = ''
	if satid:
		# reverse dictionary search
		satname = list(getISROSatelliteList().keys())[list(getISROSatelliteList().values()).index(satid)]
	return satname
	  
# search by name/satid on the dictionary         
def getSatById(sats_data, satid):
    if isinstance(satid, str):
        satid = satid.upper()
    if satid in sats_data.keys():
        return sats_data[satid]

# returns current position(km), lattitude, longitude, elvevation(m)
def getSatPosition(sats_data, satid):
	satellite = getSatById(sats_data, satid)
	# current date and time
	ts = load.timescale(builtin=True)
	t = ts.now()
	# get current pos
	geocentric = satellite.at(t)
	pos = geocentric.position.km
	# get current lat, lon, elev
	subpoint = geocentric.subpoint()
	lat = subpoint.latitude.degrees
	lon = subpoint.longitude.degrees
	elv = int(subpoint.elevation.m)
	# return current position(km), lattitude, longitude, elvevation(m)
	return pos, lat, lon, elv

# returns list of [satname, satid, lat, lon] for plotting
def getSatPositionList(sats_dict, sats_data):
	satdetails = []
	for satname, satid in sats_data.items():		
		pos, lat, lon, elv =  getSatPosition(sats_dict, satid)
		satdetails.append([satname, satid, lat,lon])
	# returns sat details
	return satdetails

# returns predicted (from currrent time) lat, lon coordinates for the duration supplied
def getSatTrackingCoord(sats_data, satid, tracking_minutes=30):
	satellite = getSatById(sats_data, satid)	
	# current date and time
	ts = load.timescale(builtin=True)
	t = ts.now()
	# default tracking for 180 minutes - 3 hrs
	minutes = np.arange(tracking_minutes)
	time    = ts.utc(t.utc.year, t.utc.month, t.utc.day, t.utc.hour, minutes)
	# get predicted lat, lon coordinates
	geocentric = satellite.at(time)
	# get current lat, lon, elev
	subpoint = geocentric.subpoint()	
	lon      = subpoint.longitude.degrees
	lat      = subpoint.latitude.degrees
	return lat, lon

# returns predicted (from currrent time) lat, lon coordinates for 
# the duration supplied (for all satellites) 
def getSatTrackingCoordList(sats_dict, sats_data, tracking_minutes=30):
	satdetails = []
	for satname, satid in sats_data.items():		
		lat, lon =  getSatTrackingCoord(sats_dict, satid, tracking_minutes)
		satdetails.append([satname, satid, lat,lon])
	# returns sat details
	return satdetails
	
# sort multiple list based on mainList indices	
def sortLists(mainList, sec1List, sec2List, sec3List):
	indices = [b[0] for b in sorted(enumerate(mainList),key=lambda i:i[1])]
	a=[]; b=[]; c=[]; d=[];
	for i in (indices):
		a.append(mainList[i])
		b.append(sec1List[i])
		c.append(sec2List[i])
		d.append(sec3List[i])
	return a, b, c, d

# plot geo satellites
def plotGEO(sd, title_text, c_latitude=10, c_longitude=80, savePlot=False):
	# get data		
	satnames=[]; satids=[]; lats=[]; lons=[]; 
	sats_details = getSatPositionList(sd, getISROGEOSatelliteList() )
	if(sats_details):
		satnames, satids, lats, lons = map(list, zip(*sats_details))
		# sort based on lon
		lons, lats, satnames, satids = sortLists(lons, lats, satnames, satids)
	# subplt
	fig, ax = plt.subplots(figsize=(10, 7))
	fig.subplots_adjust(bottom=0.01)
	fig.tight_layout()
	# projection	
	ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=c_longitude))
	ax.set_title(title_text)
	# add coastlines for reference                                                                                                
	ax.coastlines(resolution='50m')		
	ax.add_feature(cfeature.OCEAN)
	ax.set_global()
	ax.stock_img()
	# plot	
	number_of_colors = len(sats_details)
	color = ["#"+''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]	
	markers = [(i,j,0) for i in range(2,10) for j in range(1, 3)]
	tx = -80; ty = -40; dx = 20;	
	for i, (name, x, y)  in enumerate(zip(satnames, lons, lats)):
		ax.scatter( x,  y, transform=ccrs.PlateCarree(), s=30, marker=choice(markers), c=choice(color), label=name )
		ax.plot(x, y, transform=ccrs.PlateCarree(), c='r', linewidth=2 )
		# checking condition  
		if i % 2 == 0:  dy = 20; 
		else: dy = -20; 
		ax.annotate(name, xy=(x, y), xytext = (tx + dx, ty + dy), rotation=0,
					xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
					ha='right', va='top', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), 
					arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		tx = tx + dx
		if tx > 170: tx = -60; ty = -30;		
	# Put a legend below current axis	
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small', fancybox=True, shadow=True, ncol=6)
	# tropical lines
	tropics = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False, linewidth=2, linestyle='--', edgecolor='dimgrey')
	tropics.ylocator = mticker.FixedLocator([-23.43691,23.43691])
	tropics.yformatter = LATITUDE_FORMATTER
	tropics.xlines=False
	# gridlines
	gl1 = ax.gridlines(xlocs=range(-180,181,40), ylocs=range(-80,81,40),draw_labels=True)
	gl1.top_labels = False; gl1.left_labels = True
	gl2 = ax.gridlines(xlocs=range(-160,181,40), ylocs=range(-80,81,40),draw_labels=True)
	gl2.top_labels = False; gl2.left_labels = True
	# save and show
	if savePlot:
		plt.savefig('geo_tracking.png')
	plt.show()
	return

def plotTrack(sats_details, title_text, save_text='out.png', c_latitude=10, c_longitude=80, savePlot=False, tracking_minutes=30):
	if(sats_details):
		satnames, satids, lats, lons = map(list, zip(*sats_details))
	# subplt
	fig, ax = plt.subplots(figsize=(10, 7))
	fig.subplots_adjust(bottom=0.01)
	fig.tight_layout()
	# projection	
	ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=c_longitude))
	ax.set_title(title_text)
	# add coastlines for reference                                                                                                
	ax.coastlines(resolution='50m')		
	ax.add_feature(cfeature.OCEAN)
	ax.set_global()
	ax.stock_img()	
	# plot	
	number_of_colors = len(sats_details)
	color = ["#"+''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]	
	markers = [(i,j,0) for i in range(2,10) for j in range(1, 3)]
	for i, (name, x, y)  in enumerate(zip(satnames, lons, lats)):
		ax.scatter( x[0],  y[0], transform=ccrs.PlateCarree(), s=30, color='red')
		ax.text( x[0],  y[0], name, fontsize='x-small', transform=ccrs.PlateCarree()) 
		ax.plot(x, y, transform=ccrs.PlateCarree(), marker=choice(markers), c=choice(color), label=name, markersize=3, linewidth=0.7 )
	# Put a legend below current axis	
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small', fancybox=True, shadow=True, ncol=6)
	# tropical lines
	tropics = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False, linewidth=2, linestyle='--', edgecolor='dimgrey')
	tropics.ylocator = mticker.FixedLocator([-23.43691,23.43691])
	tropics.yformatter = LATITUDE_FORMATTER
	tropics.xlines=False
	# gridlines
	gl1 = ax.gridlines(xlocs=range(-180,181,40), ylocs=range(-80,81,40),draw_labels=True)
	gl1.top_labels = False; gl1.left_labels = True
	gl2 = ax.gridlines(xlocs=range(-160,181,40), ylocs=range(-80,81,40),draw_labels=True)
	gl2.top_labels = False; gl2.left_labels = True
	# save and show
	if savePlot:
		plt.savefig(save_text)
	plt.show()
	return
	
# plot leo sats
def plotLEO(sd, title, c_latitude=10, c_longitude=80, savePlot=False, tracking_minutes=30):			
	sats_details = getSatTrackingCoordList(sd, getISROLEOSatelliteList(), tracking_minutes)
	plotTrack(sats_details, title, 'leo_tracking.png', c_latitude, c_longitude, savePlot, tracking_minutes)
	return
	
# plot nav sats	
def plotNAV(sd, title, c_latitude=10, c_longitude=80, savePlot=False, tracking_minutes=600):			
	sats_details = getSatTrackingCoordList(sd, getISRONavSatelliteList(), tracking_minutes)
	plotTrack(sats_details, title, 'nav_tracking.png', c_latitude, c_longitude, savePlot, tracking_minutes)
	return
	
# plot user defined single/multiple sat tracks
def plotSatellites(sd, title, satlist, c_latitude=10, c_longitude=80, savePlot=False, tracking_minutes=300):			
	sat_details=[]	
	if(satlist):
		for satname in (satlist):
			satellite = getSatById(sd, satname)
			satid = satellite.model.satnum
			satname = getSatNameFromId(satid)
			# get lat lon
			lat, lon =  getSatTrackingCoord(sd, satid, tracking_minutes)
			# pack
			sat_details.append([satname, satid, lat,lon])
	# and plot
	plotTrack(sat_details, title, 'sats_tracking.png', c_latitude, c_longitude, savePlot, tracking_minutes)
	return	

# __main method__ 
if __name__=="__main__": 

	# create isro satellite dictionary
	sats_dict = readTLE()
	# plot center location
	c_lat=10; c_lon=80;
	# LEO plot
	tracking_minutes = 45; savePng = True;
	plotLEO(sats_dict, 'ISRO LEO satellite tracks for the next {} minutes'.format(tracking_minutes), c_lat, c_lon, savePng, tracking_minutes )
	# Nav plot
	tracking_minutes = 600; savePng = True;
	plotNAV(sats_dict, 'ISRO Navigation satellite tracks for the next {} minutes'.format(tracking_minutes), c_lat, c_lon, savePng, tracking_minutes )
	# GEO plot
	savePng = True;
	plotGEO(sats_dict, 'ISRO GEO satellites', c_lat, c_lon, savePng )	
	# Single/multiple user specified satellite tracking
	sat_names=['CARTOSAT-3','RISAT-1','RISAT-2B','RISAT-2BR1']
	tracking_minutes = 100; savePng = True;
	plotSatellites(sats_dict, 'satellite tracks for the next {} minutes'.format(tracking_minutes), sat_names, c_lat, c_lon, savePng, tracking_minutes)		

