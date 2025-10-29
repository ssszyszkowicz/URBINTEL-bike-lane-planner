from utils import *
from requests import get as requests_get


def OTPcall(OlonOlatDlonDlat, City='ottgat', Safety = 0.0, Bike_kph = 17.7, extra_params = {}):
    Olon, Olat, Dlon, Dlat = OlonOlatDlonDlat
    params = { #params described at dev.opentripplanner.org/apidoc/1.0.0/resource_PlannerResource.html
        "fromPlace":'%f,%f'%(Olat,Olon),
        "toPlace":'%f,%f'%(Dlat,Dlon),
        "bikeSpeed":Bike_kph/3.6, #must be in m/s
        #"bannedRoutes" : 'Test_Purple', # TO_DO
        "mode":'BICYCLE'}
    params.update(extra_params)
    if Safety>0.0:
        params.update({"optimize":"TRIANGLE", # (!) Must be here to trigger bicycle-triangle
        "triangleSafetyFactor":Safety,
        "triangleSlopeFactor":0.0,
        "triangleTimeFactor":1.0-Safety})
    url = "http://localhost:8080/otp/routers/"+City+"/plan?"
    url+= '&'.join([k+'='+str(v) for k,v in params.items()])
    R = json_loads(requests_get(url,timeout=25.0).text)
    Success = True
    if 'error' in R: Success = False
    else:
        I = R['plan']['itineraries']
        if len(I) != 1:
            print('Received %d itineraries! What to do?'%len(I))
            Success = False
        else:
            D = I[0]
            dur = float(D['duration'])
            dist = float(D['walkDistance'])
    if not Success:
        dur = float('NaN')
        dist = float('NaN')
    else:
        streets = []
        for L in D['legs']:
            for S in L['steps']:
                streets.append(S['streetName'])
    for var in 'IDRLS': locals().pop(var, None)
    return locals()
##    out['streets'] = {}
##    for L in D['legs']:
##        for S in L['steps']:
##            S['streetName']
##    if DISP: print(R)
##    if DISP and Success:
##        print('t[s] =',dur,'d[m] =',dist)
##        for L in D['legs']:
##            for S in L['steps']:
##                print(S['streetName']+': %dm'%int(S['distance']+0.5))
    


def OTPbatch(N_OlonOlatDlonDlat, City='ottgat', Safety = 0.0, Bike_kph = 17.7, extra_params={}):
    N = len(N_OlonOlatDlonDlat)
    out = {}
    for k in ['time_s','len_m','kph ']: out[k] = [0.0 for _ in range(N)]
    #DISP = (N==1)
    for n,OD in enumerate(N_OlonOlatDlonDlat):
        one = OTPcall(OD, City, Safety, Bike_kph, extra_params)
        out['time_s'][n] = one['dur']
        out['len_m'][n] = one['dist']
        out['kph '][n] = (one['dist']/1000.0)/(one['dur']/3600.0)        
        print(n,'...')
    return out


def GoogleAPI(OlonOlatDlonDlat):
    N = len(OlonOlatDlonDlat)
    if N > 100:
        print("SAFETY: do not send more than 100 O-D requests to Google!")
        return False
    out = {}
    for k in ['time_s','len_m','kph ']: out[k] = [0.0 for _ in range(N)]
    DISP = (N==1)
    for n,OD in enumerate(OlonOlatDlonDlat):
        Olon, Olat, Dlon, Dlat = OD
        params = { #params described at  https://developers.google.com/maps/documentation/directions/intro
            "origin":'%f,%f'%(Olat,Olon),
            "destination":'%f,%f'%(Dlat,Dlon),
            "mode":"bicycling",
            "key":"AIzaSyBbIuxPs2vW37tvJ6l53ZPm0rwCdwoEgG0"}
        url = "https://maps.googleapis.com/maps/api/directions/json?"
        url+= '&'.join([k+'='+str(v) for k,v in params.items()])

        #time_sleep(1.0+4.0*random_random())
        print("Calling Google API...")
        print(url)
        response = requests_get(url,timeout=25.0)
        R = json_loads(response.text)
        Success = True
        if 'error' in R: Success = False
        else:
            if len(R["routes"]) != 1:
                print('Received %d routes! What to do?'%len(R["routes"]))
                Success = False
            else:
                L = R["routes"][0]['legs']
                if len(L) != 1:
                    print('Received %d legs! What to do?'%len(L))
                    Success = False
                else:
                    dur = float(L[0]['duration']['value'])
                    dist = float(L[0]['distance']['value'])
        if Success == False:
            dur = float('NaN')
            dist = float('NaN')
        out['time_s'][n] = dur
        out['len_m'][n] = dist
        out['kph '][n] = (dist/1000.0)/(dur/3600.0)
        if DISP: print(L[0])
        if DISP and Success:
            print('t[s] =',dur,'d[m] =',dist)
            for S in L[0]['steps']:
                print(S['html_instructions']+' : '+S['distance']['text'])
    return out
