#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:59:09 2017

@author: giacomo
"""
import pandas as pd
import os
from dxfwrite import DXFEngine as dxf
import datetime
import math
from scipy.optimize import fsolve
import plotly
import plotly.graph_objs as go
import copy


# read CSV files in folder "coordinates" and store data into DataFrame
def readCSV(l_csv):
    df_coord=pd.DataFrame(columns=["Nome Punto","Data Misura","N","E","H"])
    for csv_file in l_csv:
        csv_file=csv_file.strip()
        # read from filename data of measurements
        data_misu_str=csv_file[:-4]
        year_str=data_misu_str[:4]
        month_str=data_misu_str[4:6]
        day_str=data_misu_str[6:]
        date=datetime.date(int(year_str),int(month_str),int(day_str))

        df=pd.read_csv(path+'/'+csv_file, names=["Nome Punto","N","E","H"],
                       header=0,decimal=',',dtype={"Nome Punto":str,"N":float ,"E":float,"H":float})
        
        # add date column to df DataFrame
        lenght=len(df["Nome Punto"])
        s_date=pd.Series(date for i in range(0,lenght))
        df.loc[:,"Data Misura"]=s_date
              
        df_coord=df_coord.append(df,ignore_index=True)
    # set "Nome Punto" column as DataFrame index
    df_coord=df_coord.set_index("Nome Punto")

    return(df_coord)
    

# define list of all points names measured
def nomiPti(df_coord):
    l_nomi_pti=df_coord.index.unique().tolist()
    l_nomi_pti.sort()
    return(l_nomi_pti)

# define list of all measure dates
def dates(df_coord):
    l_dates=df_coord["Data Misura"].unique().tolist()
    l_dates.sort()
    return(l_dates)

# define Series of zero-measure dates of all points
def datesZero(df_coord,l_nomi_pti):
    l_dates_zero=[]
    for nome_pto in l_nomi_pti:
        date_misu_i=df_coord.loc[nome_pto]["Data Misura"]
        l_dates_zero.append((sorted(date_misu_i.tolist()))[0])
    s_dates_zero=pd.Series(l_dates_zero,index=l_nomi_pti)
    return(s_dates_zero)

# calculate zero-coordinates DataFrame
def zeroCoord(df_coord,l_nomi_pti,s_dates_zero):
    l_E=[]
    l_N=[]
    l_H=[]
    for nome_pto in l_nomi_pti:
        data_zero=s_dates_zero.loc[nome_pto]
        coord_i=df_coord[df_coord["Data Misura"]==data_zero].loc[nome_pto]
        # calculation of average coordinates for each point
        E=coord_i['E'].mean()
        N=coord_i['N'].mean()
        H=coord_i['H'].mean()
        l_E.append(E)
        l_N.append(N)
        l_H.append(H)
       
    d_coord_zero={'Nome Punto':l_nomi_pti,'Data Misura':s_dates_zero.values.tolist(),'E':l_E,'N':l_N,'H':l_H}
    df_coord_zero=pd.DataFrame(d_coord_zero,columns=['Data Misura',"E","N","H"],index =l_nomi_pti)   
    return(df_coord_zero)    
    
# calculate delta DataFrame
def deltaCoord(df_coord,df_coord_zero,l_dates):
    df_coord_mean=pd.DataFrame(columns=["Data Misura","N","E","H"])
    df_delta=pd.DataFrame(columns=["Data Misura","N","E","H"])
    for date in l_dates:
        l_E=[]
        l_N=[]
        l_H=[]
        l_nomi_pti_misu=[]
        # find for each date the list of measured points names
        l_nomi_pti_misu=df_coord[df_coord["Data Misura"]==date].index.unique().tolist()
        for nome_pto in l_nomi_pti_misu:
            coord_i=df_coord[df_coord["Data Misura"]==date].loc[nome_pto]
            E=coord_i['E'].mean()
            N=coord_i['N'].mean()
            H=coord_i['H'].mean()
            l_E.append(E)
            l_N.append(N)
            l_H.append(H)
            
            
        lenght=len(l_nomi_pti_misu)
        d_coord_mean={'Nome Punto':l_nomi_pti_misu,'Data Misura':[date for i in range(0,lenght)],'E':l_E,'N':l_N,'H':l_H}
        df_coord_mean_i=pd.DataFrame(d_coord_mean,columns=['Data Misura',"E","N","H"],index =l_nomi_pti_misu)
        df_coord_mean=df_coord_mean.append(df_coord_mean_i,ignore_index=False)

    for date in l_dates:
        df_delta_i=df_coord_mean[df_coord_mean['Data Misura']==date][['E','N','H']]-df_coord_zero[['E','N','H']]
        # add date column to df DataFrame
        lenght=len(df_delta_i["E"])
        l_dates_i=[date for i in range(0,lenght)]
        s_date=pd.Series(l_dates_i,index=l_nomi_pti_misu)
        df_delta_i.loc[:,"Data Misura"]=s_date
                      
        df_delta=df_delta.append(df_delta_i,ignore_index=False)     
      
    return(df_delta,df_coord_mean)

# calculate relative delta DataFrame
def deltaCoordRel(l_dates,df_coord_mean):
    df_delta_rel=pd.DataFrame(columns=["Data Misura","N","E","H"])
    for i in range(1,len(l_dates)):
        l_deltaE=[]
        l_deltaN=[]
        l_deltaH=[]
        l_nomi_pti_misu=[]
        # find for each date the list of measured points names
        l_nomi_pti_misu=df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i]].index.tolist()
        for nome_pto in l_nomi_pti_misu:
            s_delta_rel_i=df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i]][['E','N','H']].loc[nome_pto]-df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i-1]][['E','N','H']].loc[nome_pto]
        
            deltaE=s_delta_rel_i['E']
            deltaN=s_delta_rel_i['N']
            deltaH=s_delta_rel_i['H']
            l_deltaE.append(deltaE)
            l_deltaN.append(deltaN)
            l_deltaH.append(deltaH)
                    
        lenght=len(l_nomi_pti_misu)
        d_delta_rel={'Nome Punto':l_nomi_pti_misu,'Data Misura':[l_dates[i] for j in range(0,lenght)],'E':l_deltaE,'N':l_deltaN,'H':l_deltaH}
        df_delta_rel_i=pd.DataFrame(d_delta_rel,columns=['Data Misura',"E","N","H"],index =l_nomi_pti_misu)
        df_delta_rel=df_delta_rel.append(df_delta_rel_i,ignore_index=False)
    return(df_delta_rel)

def relCoord(fs,l_dates,df_coord_mean,df_delta_rel):
    df_coord_rel=pd.DataFrame(columns=["Data Misura","N","E","H"])
    df_coord_rel=df_coord_mean[df_coord_mean['Data Misura']==l_dates[0]]
    for i in range(1,len(l_dates)):
        df_coord_rel_i=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]][['E','N','H']]+fs*df_delta_rel[df_delta_rel['Data Misura']==l_dates[i]][['E','N','H']]
        lenght=len(df_coord_rel_i["E"])
        df_coord_rel_i.loc[:,'Data Misura'] = pd.Series([l_dates[i] for j in range(0,lenght)], index=df_coord_rel_i.index)
        df_coord_rel=df_coord_rel.append(df_coord_rel_i,ignore_index=False)   
    return(df_coord_rel)       
    
# definition of list of points for each quote
def pointsLayers(l_nomi_pti):
    points_alti=['01','02','03','10','11','12']
    points_inter=['04','05','06']
    points_bassi=['07','08','09']
    return(points_alti,points_inter,points_bassi) 
    

# creation of displacement vector   
def arrowPointsCreation(x0,y0,z0,x1,y1,z1, layer_name):
    # vector magnitude calculation
    mod=math.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    # Calculation of a,d,c coefficient of the plane equation (base of arrow).
    # ax+by+cz+d=0
    # Coefficients are calculated by orthogonality condition between plane and
    # displacement vector.
    # If vector magnitude equal zero then a=b=c=0
    if mod==0.:
        a=b=c=0.
    else:
        a=(x1-x0)/mod  # coseno direttore x
        b=(y1-y0)/mod  # coseno direttore y
        c=(z1-z0)/mod  # coseno direttore z
    # Calculation of d coefficient of the plane equation.
    # Coefficient is calculated by imposing that point P belongs to plane 
    xP=x0+0.8*(x1-x0)
    yP=y0+0.8*(y1-y0)
    zP=z0+0.8*(z1-z0)
    d=-(a*xP+b*yP+c*zP)
    # Widht of arrow base.
    r=0.1*mod
    # Calculation of parameter of circumference equation to wich belong the 4
    # points of the base of the arrow.
    # x=xP+r*ex*cos(alpha)+r*fx*sin(alpha)
    # y=yP+r*ey*cos(alpha)+r*fy*sin(alpha)
    # z=zP+r*ez*cos(alpha)+r*fz*sin(alpha)
    if a==0.:
        ex=1.
    else:
        ex=-math.sqrt(a**2/(a**2+c**2))*(c/a)
    
    ey=0.
    
    if c==0.:
        ez=1.
    else:
        ez=-ex*a/c
    # Calculation of fx, fy, fz parameters imposing the condition of
    # orthogonality between e=(ex,ey,ez) and f=(fx,fy,fz)
    def equations(p):
        fx, fy, fz = p
        return (ex*fx+ey*fy+ez*fz, fx*a+fy*b+fz*c, fx**2+fy**2+fz**2-1)   
    
    fx,fy,fz = fsolve(equations, (1,1,1))
    # Alpha angles of the 4 base point of the arrow
    alpha3=0
    alpha4=math.pi/2.
    alpha5=math.pi
    alpha6=(3./2.)*math.pi
           
    # drawing of arrow points         
    P1=(x0,y0,z0)
    P2=(x1,y1,z1)
    P3=(xP+r*ex*math.cos(alpha3)+r*fx*math.sin(alpha3),yP+r*ey*math.cos(alpha3)+r*fy*math.sin(alpha3),zP+r*ez*math.cos(alpha3)+r*fz*math.sin(alpha3))
    P4=(xP+r*ex*math.cos(alpha4)+r*fx*math.sin(alpha4),yP+r*ey*math.cos(alpha4)+r*fy*math.sin(alpha4),zP+r*ez*math.cos(alpha4)+r*fz*math.sin(alpha4))
    P5=(xP+r*ex*math.cos(alpha5)+r*fx*math.sin(alpha5),yP+r*ey*math.cos(alpha5)+r*fy*math.sin(alpha5),zP+r*ez*math.cos(alpha5)+r*fz*math.sin(alpha5))
    P6=(xP+r*ex*math.cos(alpha6)+r*fx*math.sin(alpha6),yP+r*ey*math.cos(alpha6)+r*fy*math.sin(alpha6),zP+r*ez*math.cos(alpha6)+r*fz*math.sin(alpha6))
    
    # drawing of arrow lines
    lineA= dxf.line(P1, P2)
    lineA['layer'] = layer_name
    drawing.add(lineA)
    
    lineB= dxf.line(P2, P3)
    lineB['layer'] = layer_name
    drawing.add(lineB)
    
    lineC= dxf.line(P2, P4)
    lineC['layer'] = layer_name
    drawing.add(lineC)
    
    lineD= dxf.line(P2, P5)
    lineD['layer'] = layer_name
    drawing.add(lineD)
    
    lineE= dxf.line(P2, P6)
    lineE['layer'] = layer_name
    drawing.add(lineE)
    
    lineF= dxf.line(P3, P4)
    lineF['layer'] = layer_name
    drawing.add(lineF)
    
    lineG= dxf.line(P4, P5)
    lineG['layer'] = layer_name
    drawing.add(lineG)
    
    lineH= dxf.line(P5, P6)
    lineH['layer'] = layer_name
    drawing.add(lineH)
    
    lineI= dxf.line(P3, P6)
    lineI['layer'] = layer_name
    drawing.add(lineI)

# creation of labels  
def datesLabelsCreation(layer_name,i):
    # insert point of dates labels
    x0=10
    y0=-20
    z0=0
    # vertical spacing of dates labels
    delta_y=0.4
    
    text = dxf.text(layer_name, (x0,y0-(delta_y*i),z0), height=text_h)
    text['layer'] = layer_name
    drawing.add(text)
  
# creation of scale bar  
def scaleBarCreation(fs):
    # lenght of 1 cm scalebar
    fs1=fs/100
    drawing.add_layer('scale_bar', color=7)
    # insertion point of scale bar
    x0=10.
    y0=-20.
    z0=0.
    P0=(x0,y0,z0)
    P1=(x0,y0+(fs1/20.),z0)
    P2=((x0+fs1),y0,z0)
    P3=((x0+fs1),y0+(fs1/20.),z0)
       
    line1= dxf.line(P0,P1)
    line1['layer'] = 'scale_bar'
    drawing.add(line1)
    
    line2= dxf.line(P0,P2)
    line2['layer'] = 'scale_bar'
    drawing.add(line2)
    
    line3= dxf.line(P2,P3)
    line3['layer'] = 'scale_bar'
    drawing.add(line3)
    
    text = dxf.text('0', P1, height=text_h)
    text['layer'] = 'scale_bar'
    drawing.add(text)
    
    text = dxf.text('1 cm', P3, height=text_h)
    text['layer'] = 'scale_bar'
    drawing.add(text)

# creation of points names labels
def pointsLabels(df_coord_zero):
    
    for nome_pto in df_coord_zero.index:
        x=df_coord_zero['E'].loc[nome_pto]
        y=df_coord_zero['N'].loc[nome_pto]
        z=df_coord_zero['H'].loc[nome_pto]
        
        drawing.add_layer('points_names', color=7)
        text = dxf.text(nome_pto, (x, y,z), height=text_h)
        text['layer'] = 'points_names'
        drawing.add(text)
    
# creation of displacements arrows
def drawDisp(l_dates,df_coord_zero,df_coord_rel):
    
    j=1 # layer color index
    
    for i in range(1,len(l_dates)):
        layer_name=l_dates[i].strftime('%d%m%Y')
        drawing.add_layer(layer_name, color=j)
        
        datesLabelsCreation(layer_name,i)

        l_nomi_pti_misu=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].index.tolist()
        
        for nome_pto in l_nomi_pti_misu:

            x0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['E']
            y0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['N']
            z0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['H']
         
            x1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['E']
            y1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['N']
            z1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['H']
            
            arrowPointsCreation(x0,y0,z0,x1,y1,z1,layer_name)

        j=j+1
    drawing.save()
    
# creation of time-delta graphs
def graphTD(df_delta, l_nomi_pti):
    # inizializzazione variabili
    traces=[]
    l_visible=[False for i in range(0,3*len(l_nomi_pti))]
    l_buttons=[]
    i=0
    # per ogni punto misurato vengono presi i dati da graficare dalla variabile
    # di tipo DataFrame 'df_delta'
    for nome_pto in l_nomi_pti:
        # definizione 'traces' da rappresentare in asse x e y
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['E'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'x'
                    )
                )            
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['N'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'y'
                    )
                )
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['H'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'z'
                    )
                )
        # definizione del menù a tendina per la selezione della serie da rappresentare
        # nella posizione i-esima della lista 'l_visible_i' viene scritto True
        l_visible_i=copy.deepcopy(l_visible)
        l_visible_i[i]=True
        l_visible_i[i+1]=True
        l_visible_i[i+2]=True
                   
        l_buttons.append({
          "args": ["visible", l_visible_i], 
          "label": nome_pto, 
          "method": "restyle"
          })
        
        i=i+3
        
    data = go.Data(traces)
    # definizione delle caratteristiche dell'asse y
    yaxis_par=dict(title='spostamenti [m]',
                   titlefont=dict(family='Arial', size=12, color='black'),
                   showticklabels=True,
                   tickangle=0,
                   tickfont=dict(family='Arial',size=12,color='black'),
                   range=[-0.04, 0.04])
    # impostazione del layout del grafico
    layout={'title':'Grafico spostamenti',
            'yaxis': yaxis_par,
            'updatemenus':[{'x':-0.05,'y':1,'yanchor':'top','buttons':l_buttons}]
            }
            
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig,filename='graph_TD.html')
    
# creation of list of mutual height distance for each vertical section
def dist_alt(df_coord_zero, l_allin):
   
    d_2_1=[]
    
    for i in range(0,len(l_allin)-1):    
        coord_H_1 = df_coord_zero.loc[l_allin[i]].H
        coord_H_2 = df_coord_zero.loc[l_allin[i+1]].H
        d_2_1.append(coord_H_2-coord_H_1)  
    # calcolo distanze cumulate per asse x grafico
    d_cum = [0]
    for i in range(0,len(d_2_1)):
        d_cum.append(d_cum[i]+d_2_1[i])
    
    return d_cum   

# creation of time-delta graphs of vertical sections
def graphSection(df_delta, d_cum, l_vert, filename_i):
    # inizializzazione variabili
    traces=[]
    l_visible=[False for i in range(0,len(df_delta['Data Misura'].unique()))]
    l_buttons=[]
    i=0
       
    # per ogni punto misurato vengono presi i dati da graficare dalla variabile
    # di tipo DataFrame 'df_delta'
    for data_misu in df_delta['Data Misura'].unique():
        # definizione 'traces' da rappresentare in asse x e y
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta[df_delta['Data Misura']==data_misu].loc[l_vert].E,
                y = d_cum,               
                mode = 'lines',
                # nome della serie di dati
                name = data_misu
                    )
                )            

        # definizione del menù a tendina per la selezione della serie da rappresentare
        # nella posizione i-esima della lista 'l_visible_i' viene scritto True
        l_visible_i=copy.deepcopy(l_visible)
        l_visible_i[i]=True
                
        l_buttons.append({
          "args": ["visible", l_visible_i], 
          "label": data_misu, 
          "method": "restyle"
          })
        
        i=i+1
        
    data = go.Data(traces)
    # definizione delle caratteristiche dell'asse y
    yaxis_par=dict(title='nome punto',
                   titlefont=dict(family='Arial', size=12, color='black'),
                   showticklabels=True,
                   tickvals=d_cum,    # posizioni delle etichette dell'asse y mostrate
                   ticktext=l_vert, # definizione delle etichette dell'asse y
                   tickangle=0,
                   tickfont=dict(family='Arial',size=12,color='black'),
                   autorange=True
                   )
    
    # definizione delle caratteristiche dell'asse y
    xaxis_par=dict(title='spostamenti in dir. normale [m]',
                   titlefont=dict(family='Arial', size=12, color='black'),
                   showticklabels=True,
                   tickangle=0,
                   tickfont=dict(family='Arial',size=12,color='black'),
                   range=[-0.01, 0.03]
                   )
    # impostazione del layout del grafico
    layout={'title':'Grafico spostamenti in direzione normale alla paratia',
            'yaxis': yaxis_par,
            'xaxis': xaxis_par
            #'updatemenus':[{yanchor':'top','buttons':l_buttons}]
            }
            
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig,filename=filename_i)
    #plotly.offline.plot(fig,image='png',image_width=1600, image_height=1200)

#------------------------------------------------------------------------------

path="./coordinate"

# height of text labels
text_h=0.2
# scale factor for dxf rappresentation
fs = 100
l_csv=os.listdir(path)
df_coord=readCSV(l_csv)
l_nomi_pti=nomiPti(df_coord)
l_dates=dates(df_coord)
s_dates_zero=datesZero(df_coord,l_nomi_pti)
df_coord_zero=zeroCoord(df_coord,l_nomi_pti,s_dates_zero)
[df_delta,df_coord_mean]=deltaCoord(df_coord,df_coord_zero,l_dates)
df_delta_rel=deltaCoordRel(l_dates,df_coord_mean)
df_coord_rel=relCoord(fs,l_dates,df_coord_mean,df_delta_rel)

# creation of dxf file of all dispacements
drawing= dxf.drawing('paratia.dxf')
pointsLabels(df_coord_zero)
scaleBarCreation(fs)
drawDisp(l_dates,df_coord_zero,df_coord_rel)

# creation of plotly time-displacements graphs
graphTD(df_delta, l_nomi_pti)

# creation of plotly normal-displacements graphs
# definizioni dei punti da rappresentare per ogni verticale
l_vert_1=['07','04','01']
filename_1='graph_vert_1.html'
l_vert_2=['08','05','02']
filename_2='graph_vert_2.html'
l_vert_3=['09','06','03']
filename_3='graph_vert_3.html'
# calcolo mutue distanze fra punti
d_cum_1=dist_alt(df_coord_zero, l_vert_1)
d_cum_2=dist_alt(df_coord_zero, l_vert_2)
d_cum_3=dist_alt(df_coord_zero, l_vert_3)
# plotly normal-displacements graphs
graphSection(df_delta, d_cum_1, l_vert_1, filename_1)
graphSection(df_delta, d_cum_2, l_vert_2, filename_2)
graphSection(df_delta, d_cum_3, l_vert_3, filename_3)

# write DataFrames to csv
df_coord.to_csv('coordinate.csv',sep='\t',decimal=',',float_format='%.3f')
df_delta.to_csv('spostamenti.csv',sep='\t',decimal=',',float_format='%.3f')


'''
# creation of time-delta graphs
def graphTDImage(df_delta, l_nomi_pti):
    # inizializzazione variabili
    traces=[]
    l_buttons=[]
    i=0
    
    # definizione delle caratteristiche dell'asse y
    yaxis_par=dict(title='spostamenti [m]',
                   titlefont=dict(family='Arial', size=20, color='black'),
                   showticklabels=True,
                   tickangle=0,
                   tickfont=dict(family='Arial',size=20,color='black'),
                   range=[-0.04, 0.04])
    # impostazione del layout del grafico
    layout=go.Layout(title='Grafico spostamenti',
            font=dict(family='Arial', size=20, color='black'),
            yaxis=yaxis_par,
            updatemenus=[{'x':-0.05,'y':1,'yanchor':'top','buttons':l_buttons}]
            )
    
    # per ogni punto misurato vengono presi i dati da graficare dalla variabile
    # di tipo DataFrame 'df_delta'
    for nome_pto in l_nomi_pti:
        traces=[]
        # definizione 'traces' da rappresentare in asse x e y
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['E'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'E'
                    )
                )            
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['N'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'N'
                    )
                )
        traces.append(plotly.graph_objs.Scatter(
                x = df_delta.loc[nome_pto]['Data Misura'],
                y = df_delta.loc[nome_pto]['H'],
                mode = 'lines',
                # nome della serie di dati
                name = nome_pto + 'H'
                    )
                )
                   
        l_buttons.append({
          "args": ["visible", l_visible_i], 
          "label": nome_pto, 
          "method": "restyle"
          })
        
        data = go.Data(traces)
        
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig,image='png',image_width=1600, image_height=1200)
        
        i=i+3
        
'''
            
