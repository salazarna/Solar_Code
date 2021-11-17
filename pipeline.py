import pvlib
import numpy as np
import pandas as pd
import cnosolar as cno
from tqdm.auto import tqdm
from functools import reduce

def run(system_configuration, data, irrad_instrument, availability, energy_units):
    '''
    Wrapper that executes the production stages of the PV system, 
    including system losses.
    
    Parameters
    ----------
    system_configuration : list
        List of system configuration .JSON files in dict format. If the PV 
        plant consists of multiple configuration .JSON files in dict format, 
        they are arranged alphabetically.

    data : pandas.DataFrame
        Historical series of meteorological data. The data structure follows 
        the established one by CREG 060 of 2019, i.e., time stamp, :math:`GHI` 
        and :math:`Tamb` if the parameters :math:`POA` and :math:`Tmod` are added, 
        they prevail for the calculations of the algorithms (e.g., decomposition 
        and transposition models are not used to determine :math:`POA` or 
        temperature models to determine :math:`Tmod`).
        
    irrad_instrument : string
        Indicate the instrument with which the POA irradiance measurements were obtained. 
        This parameter is used to estimate the effective irradiance. Valid options are
        'Piranómetro' and 'Celda de Referencia'.

    availability : list
        Percentage value of availability per inverter set with the exact 
        same electrical configuration.
        Default = 1.0

    energy_units : string
        Energy units to scale the calculations. Used to allow adaptation 
        of the energy results report.

    Returns
    -------
    bus_pipeline : dict
        Data structure that contains the following parameters:
        1. location - PVlib Location defined class.
        2. solpos - Data structure that contains solar zenith and solar azimuth in [degrees].
        3. airmass - Data structure that contains unitless relative and absolute airmass.
        4. etr_nrel - Extraterrestrial radiation from time stamps of the historical 
                      data series in [W/m2].
        5. disc - Data structure that contains the following parameters:
                      1. dni - Modeled direct normal irradiance provided by the Direct 
                               Insolation Simulation Code (DISC) model in [W/m2].
                      2. kt - Ratio of global to extraterrestrial irradiance on a 
                              horizontal plane.
                      3. airmass - Airmass.
                      4. dhi - Diffuse horizontal irradiance calculated by the fraction
                               of the difference of GHI and DNI, and the cosine of 
                               solar zenith in [W/m2].
        6. tracker - Data structure that contains the following parameters:
                         1. tracker_theta - Rotation angle of the tracker (zero is horizontal, and 
                                            positive rotation angles are clockwise) in [degrees].
                         2. aoi - Angle-of-incidence of DNI onto the rotated panel surface 
                                  in [degrees].
                         3. surface_tilt - Angle between the panel surface and the earth 
                                           surface, accounting for panel rotation in [degrees].
                         4. surface_azimuth - Azimuth of the rotated panel, determined by 
                                              projecting the vector normal to the panel’s surface 
                                              to the earth’s surface, in [degrees].
        7. mount - PVlib Mount defined class.
        8. bifacial - Parameter that checks if the PV modules are bifacial.
        9. total_incident_front - Total incident front irradiance if PV module is bifacial
                                  in [W/m2]. 
        10. total_incident_back - Total incident rear irradiance if PV module is bifacial
                                  in [W/m2].
        11. total_absorbed_front - Total absorbed front irradiance if PV module is bifacial,
                                   taking into account spectral and mismatch losses, in [W/m2].
        12. total_absorbed_back - Total absorbed rear irradiance if PV module is bifacial,
                                  taking into account spectral and mismatch losses, in [W/m2].
        13. poa - Effective plane-of-array irradiance, taking into account spectral and 
                  mismatch losses, in [W/m2].
        14. temp_cell - Average cell temperature of cells within a module in [ºC].
        15. dc - Data structure that contains the following parameters:
                     1. i_sc - Short circuit current in [A].
                     2. v_oc - Open circuit voltage in [V].
                     3. i_mp -Current at maximum power point in [A].
                     4. v_mp -Voltage at maximum power point in [V].
                     5. p_mp -Power at maximum power point in [W].
                     6. i_x -Current at V=0.5·Voc in [A].
                     7. i_xx -Current at V=0.5·(Voc+Vmp) in [A].
        16. ac - AC power output in [W].
        17. energy - Data structure that contains the following parameters:
                         1. Daily energy in selected units. Default units in [Wh].
                         2. Weekly energy in selected units. Default units in [Wh].
                         3. Monthly energy in selected units. Default units in [Wh].

    Notes
    -----
    The calculation procedure is:
        1. Define a pvlib.location Location class and estimate the solar position 
           parameters, airmass and extraterrestrial DNI.
        2. Define a pvlib.pvsystem Mount class and determine the surface orientation
           if mount is in a fixed tilt or module orientation if the mount is in a 
           single axis tracker.
        3. Determine the POA irradiance using DISC decomposition and Perez-Ineichen 1990
           transposition models if GHI is provided or leaving the supplied values in the 
           historical series of meteorological data if POA is provided.
        3. Determine the Spectral Mismatch Modifier to calculate the effective irradiance.
        4. Calculate effective POA irradiance as the product of Spectral Mismatch Modifier,
           POA irradiance, cosine of angle-of-incidence (AOI) and incidence angle modifier
           (IAM). Does not apply if POA is provided in the historical series of meteorological
           data and the irrad_instrument is 'Celda de Referencia'.
        5. Calculate total and absorbed front, and total and absorbed back irradiance if
           the module is bifacial.
        6. Define a pvlib.pvsystem Array class.
        7. Define a pvlib.pvsystem PVSystem class.
        8. Determine the average cell temperature of cells within a module using TNOCT model
           if :math:`Tmod` is not provided in the historical series of meteorological data.
        9. Calculate the PV system production, including system losses.
        10. Generate a full simulation results report per PV system subarrays and for 
            the inverter, by adding the subarrays production.
      
    See also
    --------
    cno.location_data.get_parameters
    cno.irradiance_models.decomposition
    cno.irradiance_models.transposition
    cno.pvstructure.get_mount_tracker
    cno.cell_temperature.from_tnoct
    cno.production.dc_production
    cno.production.losses
    cno.production.ac_production_sandia
    cno.production.ac_production_pvwatts
    cno.production.get_energy
    '''
    bus_pipeline = {}
    num_systems = len(system_configuration)
    
    resolution = data.index.to_series().diff().median().total_seconds()/60
    
    if availability == None:
        inv_availability = list(np.repeat(1, num_systems))
    else:
        inv_availability = availability

    for j in tqdm(range(num_systems), desc='Sistema/Inversor (.JSON)', leave=False):
        sc = system_configuration[j]
        num_subarrays = sc['num_arrays']
        
        if num_systems > 1:
            superkey = f'inverter{j+1}'
        else:
            superkey = 'plant'
            
        bus_pipeline[superkey] = {}
        
        for i in tqdm(range(num_subarrays), desc='Subarrays', leave=False):
            # Meteorological Data
            location, solpos, airmass, etr_nrel = cno.location_data.get_parameters(latitude=sc['latitude'], 
                                                                                   longitude=sc['longitude'], 
                                                                                   tz=sc['tz'], 
                                                                                   altitude=sc['altitude'], 
                                                                                   datetime=data.index)

            # Mount and Tracker
            if sc['with_tracker'] == False:
                sur_tilt = sc['surface_tilt'][i]
                sur_azimuth = sc['surface_azimuth'][i]
                ax_tilt = None
                ax_azimuth = None
                m_angle = None
            else:
                sur_tilt = None
                sur_azimuth = None
                ax_tilt = sc['axis_tilt'][i]
                ax_azimuth = sc['axis_azimuth'][i]
                m_angle = sc['max_angle'][i]

            mount, tracker = cno.pvstructure.get_mount_tracker(with_tracker=sc['with_tracker'], 
                                                               surface_tilt=sur_tilt, 
                                                               surface_azimuth=sur_azimuth, 
                                                               solpos=solpos, 
                                                               axis_tilt=ax_tilt, 
                                                               axis_azimuth=ax_azimuth, 
                                                               max_angle=m_angle,
                                                               racking_model=sc['racking_model'])
            
            # Spectral Mismatch
            if sc['with_tracker'] == False:
                st = list(np.repeat(sur_tilt, len(data)))
                sa = list(np.repeat(sur_azimuth, len(data)))
                aoi = pvlib.irradiance.aoi(surface_tilt=st,
                                           surface_azimuth=sa, 
                                           solar_zenith=solpos.apparent_zenith, 
                                           solar_azimuth=solpos.azimuth)
            else:
                st = tracker.surface_tilt
                sa = tracker.surface_azimuth
                aoi = tracker.aoi

            iam = pvlib.iam.physical(aoi=aoi, n=1.526, K=4.0, L=0.002)
            
            ## Precipitable Water
            pw = pvlib.atmosphere.gueymard94_pw(temp_air=data['Tamb'], 
                                                relative_humidity=sc['relative_humidity'])

            t = sc['module']['Technology']
            if t in ['Mono-c-Si', 'mc-Si', 'c-Si', 'monoSi', 'monosi', 'xsi', 'Thin Film', 'Si-Film', 'HIT-Si', 'EFG mc-Si']:
                module_tec = 'monosi'
            elif t in ['Multi-c-Si', 'multiSi', 'polySi', 'multisi', 'polysi', 'mtSiPoly']:
                module_tec = 'multisi'
            elif t in ['CIGS', 'CIS', 'cis', 'cigs']:
                module_tec = 'cigs'
            elif t in ['CdTe', 'CdTe', 'cdte', 'GaAs']:
                module_tec = 'cdte'
            elif t in ['asi', 'amorphous', 'a-Si / mono-Si', '2-a-Si', '3-a-Si']:
                module_tec = 'asi'
            else:
                module_tec = None
            
            ## Spectral Mismatch Modifier
            sm = pvlib.atmosphere.first_solar_spectral_correction(pw=pw, 
                                                                  airmass_absolute=airmass.airmass_absolute, 
                                                                  module_type=module_tec,
                                                                  coefficients=None)
            spectral_mismatch = sm.fillna(1)
            
            # POA/Effective Irradiance
            if 'POA' in list(data.columns):
                disc = pd.DataFrame(data={'disc': list(np.repeat(0, len(data)))}, index=data.index)
                poa = data['POA'] # Assumed as effective irradiance if irrad_instrument == 'Celda de Referencia'

                # Effective Irradiance
                if irrad_instrument == 'Piranómetro':
                    poa = spectral_mismatch * abs(poa * np.cos(aoi) * iam)
            
            else:
                # Decomposition
                disc = cno.irradiance_models.decomposition(ghi=data['GHI'], 
                                                           solpos=solpos, 
                                                           datetime=data.index) 

                # Transposition
                poa = cno.irradiance_models.transposition(with_tracker=sc['with_tracker'], 
                                                          tracker=tracker, 
                                                          surface_tilt=sur_tilt,
                                                          surface_azimuth=sur_azimuth,
                                                          solpos=solpos, 
                                                          disc=disc, 
                                                          ghi=data['GHI'],
                                                          etr_nrel=etr_nrel, 
                                                          airmass=airmass,
                                                          surface_albedo=sc['surface_albedo'],
                                                          surface_type=sc['surface_type'])
                
                # Effective Irradiance
                poa = spectral_mismatch * (abs(poa['poa_direct'] * np.cos(aoi) * iam + poa['poa_diffuse']))

            # Total Bifacial Effective Irradiance
            if sc['bifacial'] == True:
                # Irradiance Components
                if 'POA' in list(data.columns):
                    irrad_components = cno.irradiance_models.decomposition(ghi=data['GHI'], 
                                                                           solpos=solpos, 
                                                                           datetime=data.index) 

                    bifacial_dni = irrad_components.dni
                    bifacial_dhi = irrad_components.dhi
                else:
                    bifacial_dni = disc.dni
                    bifacial_dhi = disc.dhi            
                
                # Axis Azimuth Parameter
                if sc['with_tracker'] == False:
                    axis_azimuth = sur_azimuth + 90
                else:
                    axis_azimuth = ax_azimuth

                # Bifacial Irradiance
                bifacial_irrad = pvlib.bifacial.pvfactors_timeseries(solar_azimuth=solpos.azimuth, 
                                                                     solar_zenith=solpos.apparent_zenith, 
                                                                     surface_azimuth=sa, 
                                                                     surface_tilt=st, 
                                                                     axis_azimuth=axis_azimuth, 
                                                                     timestamps=data.index, 
                                                                     dni=bifacial_dni, 
                                                                     dhi=bifacial_dhi, 
                                                                     gcr=2.0/7.0,
                                                                     pvrow_height=sc['row_height'],
                                                                     pvrow_width=sc['row_width'],
                                                                     albedo=sc['surface_albedo'], 
                                                                     n_pvrows=3,
                                                                     index_observed_pvrow=1,
                                                                     rho_front_pvrow=0.03,
                                                                     rho_back_pvrow=0.05,
                                                                     horizon_band_angle=15.0)
                
                total_incident_front = bifacial_irrad[0]
                total_incident_back = bifacial_irrad[1]
                total_absorbed_front = bifacial_irrad[2]
                total_absorbed_back = bifacial_irrad[3]
                is_bifacial = True
                
                # Total Effective Irradiance
                poa = spectral_mismatch * (poa + (sc['bifaciality']*bifacial_irrad[3])) # bifacial_irrad[2] instead of (poa + )
            
            else:
                total_incident_front = None
                total_incident_back = None
                total_absorbed_front = None
                total_absorbed_back = None
                is_bifacial = False
            
            # Arrays
            string_array = cno.def_pvsystem.get_arrays(mount=mount,
                                                       surface_albedo=sc['surface_albedo'],
                                                       surface_type=sc['surface_type'], 
                                                       module_type=sc['module_type'], 
                                                       module=sc['module'], 
                                                       mps=sc['modules_per_string'][i], 
                                                       spi=sc['strings_per_inverter'][i])

            # PV System
            system = cno.def_pvsystem.get_pvsystem(with_tracker=sc['with_tracker'], 
                                                   tracker=tracker, 
                                                   string_array=string_array, 
                                                   surface_tilt=sur_tilt, 
                                                   surface_azimuth=sur_azimuth,
                                                   surface_albedo=sc['surface_albedo'],
                                                   surface_type=sc['surface_type'], 
                                                   module_type=sc['module_type'], 
                                                   module=sc['module'], 
                                                   inverter=sc['inverter'], 
                                                   racking_model=sc['racking_model'])
            
            # Cell Temperature
            if 'Tmod' in list(data.columns):
                temp_cell = data['Tmod']
            
            else:
                temp_cell = cno.cell_temperature.from_tnoct(poa=poa, 
                                                            temp_air=data['Tamb'], 
                                                            tnoct=sc['module']['T_NOCT'])

            # DC Production, AC Power and Energy
            dc, ac, energy = cno.production.production_pipeline(poa=poa, 
                                                                cell_temperature=temp_cell, 
                                                                module=sc['module'], 
                                                                inverter=sc['inverter'], 
                                                                system=system, 
                                                                ac_model=sc['ac_model'], 
                                                                loss=sc['loss'], 
                                                                resolution=resolution, 
                                                                num_inverter=sc['num_inverter'],
                                                                per_mppt=sc['per_mppt'][i],
                                                                availability=inv_availability[j],
                                                                energy_units=energy_units)
            # Bus Pipeline Dataframe
            if num_subarrays > 1:
                key = f'subarray{i+1}'
            else:
                key = 'system'
            
            bus_pipeline[superkey][key] = {'location': location, 
                                           'solpos': solpos, 
                                           'airmass': airmass, 
                                           'etr_nrel': etr_nrel,
                                           'disc': disc,
                                           'tracker': tracker, 
                                           'mount': mount,
                                           'bifacial': is_bifacial,
                                           'total_incident_front': total_incident_front,
                                           'total_incident_back': total_incident_back,
                                           'total_absorbed_front': total_absorbed_front,
                                           'total_absorbed_back': total_absorbed_back,
                                           'poa': poa,
                                           'string_array': string_array,
                                           'system': system,
                                           'temp_cell': temp_cell,
                                           'dc': dc, 
                                           'ac': ac, 
                                           'energy': energy}
    
        # AC and Energy Adition for Inverter
        if num_subarrays > 1:
            ac_string = []
            denergy_string = []
            wenergy_string = []
            menergy_string = []
            
            for i in range(num_subarrays):
                ac_string.append(bus_pipeline[superkey][f'subarray{i+1}']['ac'])
                denergy_string.append(pd.DataFrame(bus_pipeline[superkey][f'subarray{i+1}']['energy']['day']).energy)
                wenergy_string.append(pd.DataFrame(bus_pipeline[superkey][f'subarray{i+1}']['energy']['week']).energy)
                menergy_string.append(pd.DataFrame(bus_pipeline[superkey][f'subarray{i+1}']['energy']['month']).energy)

            sys_ac = reduce(lambda a, b: a.add(b, fill_value=0), ac_string)
            sys_denergy = reduce(lambda a, b: a.add(b, fill_value=0), denergy_string)
            sys_wenergy = reduce(lambda a, b: a.add(b, fill_value=0), wenergy_string)
            sys_menergy = reduce(lambda a, b: a.add(b, fill_value=0), menergy_string)
                
            bus_pipeline[superkey]['system'] = {'location': location, 
                                                'solpos': solpos, 
                                                'airmass': airmass, 
                                                'etr_nrel': etr_nrel,
                                                'disc': disc,
                                                'tracker': tracker, 
                                                'mount': mount,
                                                'bifacial': is_bifacial,
                                                'total_incident_front': total_incident_front,
                                                'total_incident_back': total_incident_back,
                                                'total_absorbed_front': total_absorbed_front,
                                                'total_absorbed_back': total_absorbed_back,
                                                'poa': poa,
                                                'temp_cell': temp_cell,
                                                'dc': dc, 
                                                'ac': sys_ac, 
                                                'energy': {'day': sys_denergy,
                                                           'week': sys_wenergy,
                                                           'month': sys_menergy}}
            
            
    # AC and Energy Adition for System
    if num_systems > 1:
        ac_inv = []
        denergy_inv = []
        wenergy_inv = []
        menergy_inv = []

        for i in range(num_systems):
            ac_inv.append(bus_pipeline[f'inverter{i+1}']['system']['ac'])
            denergy_inv.append(pd.DataFrame(bus_pipeline[f'inverter{i+1}']['system']['energy']['day']).energy)
            wenergy_inv.append(pd.DataFrame(bus_pipeline[f'inverter{i+1}']['system']['energy']['week']).energy)
            menergy_inv.append(pd.DataFrame(bus_pipeline[f'inverter{i+1}']['system']['energy']['month']).energy)

        sys_ac = reduce(lambda a, b: a.add(b, fill_value=0), ac_inv)
        sys_denergy = reduce(lambda a, b: a.add(b, fill_value=0), denergy_inv)
        sys_wenergy = reduce(lambda a, b: a.add(b, fill_value=0), wenergy_inv)
        sys_menergy = reduce(lambda a, b: a.add(b, fill_value=0), menergy_inv)

        bus_pipeline['plant'] = {'location': location, 
                                 'solpos': solpos, 
                                 'airmass': airmass, 
                                 'etr_nrel': etr_nrel,
                                 'disc': disc,
                                 'tracker': tracker, 
                                 'mount': mount,
                                 'bifacial': is_bifacial,
                                 'total_incident_front': total_incident_front,
                                 'total_incident_back': total_incident_back,
                                 'total_absorbed_front': total_absorbed_front,
                                 'total_absorbed_back': total_absorbed_back,
                                 'poa': poa,
                                 'temp_cell': temp_cell,
                                 'dc': dc, 
                                 'ac': sys_ac, 
                                 'energy': {'day': sys_denergy,
                                            'week': sys_wenergy,
                                            'month': sys_menergy}}
    
    return bus_pipeline