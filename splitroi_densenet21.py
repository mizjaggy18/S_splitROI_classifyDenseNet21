from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser
import logging
import shutil

import os
import numpy as np
from shapely.geometry import shape, box, Polygon, Point, MultiPolygon, LineString
from shapely import wkt
from shapely.ops import split

from glob import glob

import cytomine
from cytomine import Cytomine, CytomineJob
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection, Job, JobData, TermCollection, ImageInstanceCollection
import torch
from torchvision.models import DenseNet

import cv2
import math

from shapely import affinity
from shapely.geometry.multipolygon import MultiPolygon
from scipy.spatial import Voronoi



__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "2.1.0"
# Date created: 20 Oct 2022

def _quadrat_cut_geometry(geometry, quadrat_width, min_num=3):
    """
    Split a Polygon or MultiPolygon up into sub-polygons of a specified size.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        the geometry to split up into smaller sub-polygons
    quadrat_width : numeric
        the linear width of the quadrats with which to cut up the geometry (in
        the units the geometry is in)
    min_num : int
        the minimum number of linear quadrat lines (e.g., min_num=3 would
        produce a quadrat grid of 4 squares)

    Returns
    -------
    geometry : shapely.geometry.MultiPolygon
    """
    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = geometry.bounds
    x_num = math.ceil((east - west) / quadrat_width) + 1
    y_num = math.ceil((north - south) / quadrat_width) + 1
    x_points = np.linspace(west, east, num=max(x_num, min_num))
    y_points = np.linspace(south, north, num=max(y_num, min_num))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points]
    lines = vertical_lines + horizont_lines

    # recursively split the geometry by each quadrat line    
    for line in lines:
        geometry = MultiPolygon(split(geometry, line))

    return geometry

#============================================

def save_annotations (output,id_image,id_project,id_term_poly):
    annotations = AnnotationCollection()
    # print("Annotation collections: ", annotations)
    for i, annotation_poly in enumerate(output): 
        annotations.append(Annotation(location=annotation_poly.wkt,
                                    id_image=id_image,
                                    id_project=id_project,
                                    id_terms=[id_term_poly]))
        print(".",end = '',flush=True)
        # if i == 2000 or i == 4000 or i == 6000 or i == 8000 or i == 10000 or i == 12000 or i == 14000 or i == 16000 or i == 18000 or i == 20000 or i == 22000 or i == 24000 or i == 26000 or i == 28000 or i == 30000: 
        #     print("i==",i)
        #     annotations.save(chunk=1000)
        #     annotations = AnnotationCollection()
    annotations.save(chunk=1000)


def run(cyto_job, parameters):
    logging.info("----- ROI Split-poly v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project

    job.update(status=Job.RUNNING, progress=10, statusComment="Initialization...")

    modelname = "66k-4classnpc_densenet21adam_best_model_100ep.pth"    
    gpuid = 0
    device = torch.device(gpuid if gpuid!=-2 and torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    checkpoint = torch.load(modelname, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    print("Model name: ",modelname)
    print(f"Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    
    print(terms)
    for term in terms:
        print("ID: {} | Name: {}".format(
            term.id,
            term.name
        )) 
    job.update(status=Job.RUNNING, progress=20, statusComment="Terms collected...")
    

    images = ImageInstanceCollection().fetch_with_filter("project", project.id)
    
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = [int(id_img) for id_img in parameters.cytomine_id_images.split(',')]
        print('Images: ', list_imgs)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")
             
    id_project = parameters.cytomine_id_project
    id_term = parameters.cytomine_id_roi_term
    id_term_poly = parameters.cytomine_id_roipoly_term
    poly_sides = parameters.cytomine_poly_sides
    
    working_path = os.path.join("tmp", str(job.id))
    
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:
        for id_image in list_imgs:
            print('Parameters (id_project, id_image, id_term, id_term_poly):',id_project, id_image, id_term, id_term_poly)

            roi_annotations = AnnotationCollection()
            roi_annotations.project = id_project
            roi_annotations.image = id_image
            roi_annotations.term = id_term
            roi_annotations.showWKT = True
            roi_annotations.showMeta = True
            roi_annotations.showGIS = True
            roi_annotations.showTerm = True
            roi_annotations.fetch()
            print(roi_annotations)

            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0
            id_terms = 0

            job.update(status=Job.RUNNING, progress=40, statusComment="Running splitpoly on ROI-WSI...")

            for i, roi in enumerate(roi_annotations):
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
                print("----------------------------Split Poly------------------------------")
                roi_geometry = wkt.loads(roi.location)
                # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)
                minx=roi_geometry.bounds[0]
                maxx=roi_geometry.bounds[2]
                miny=roi_geometry.bounds[1]
                maxy=roi_geometry.bounds[3]
                widthROI=maxx-minx
                heightROI=maxy-miny
                #Dump ROI image into local PNG file
                # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                print(roi_path)
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                print("roi_png_filename: %s" %roi_png_filename)
                # roi.dump(dest_pattern=roi_png_filename)
                # im=Image.open(roi_png_filename)

                if poly_sides <= 512: 
                    if widthROI > 10000 or heightROI > 10000:
                        print("Large ROI")
                        largepoly_sides=4096
                        job.update(status=Job.RUNNING, progress=50, statusComment="Split ROI-WSI into 4096px sides...")
                        
                        outputlarge = _quadrat_cut_geometry(roi_geometry, quadrat_width=largepoly_sides, min_num=1)
                        outputlarge_poly = len(list(outputlarge))
                        print("Output large polygons: ", outputlarge_poly)

                        for i, roi_large in enumerate(outputlarge):
                            print("Current ROI4096:", str(i), "out of: ", str(outputlarge_poly))
                            job.update(status=Job.RUNNING, progress=60, statusComment="Split ROI-WSI into user-defined sides...")

                            output = _quadrat_cut_geometry(roi_large, quadrat_width=poly_sides, min_num=1)
                            output_poly = list(output)
                            print("Output polygons: ", len(output_poly))                                
                            save_annotations (output,id_image,id_project,id_term_poly)

                    else:
                        output = _quadrat_cut_geometry(roi_geometry, quadrat_width=poly_sides, min_num=1)
                        output_poly = list(output)
                        print("Output polygons: ", len(output_poly))
                        save_annotations (output,id_image,id_project,id_term_poly)

                else:
                    output = _quadrat_cut_geometry(roi_geometry, quadrat_width=poly_sides, min_num=1)
                    output_poly = list(output)
                    print("Output polygons: ", len(output_poly))
                    save_annotations (output,id_image,id_project,id_term_poly)
                


            splitroiannotations = AnnotationCollection()
            splitroiannotations.project = id_project
            splitroiannotations.image = id_image
            splitroiannotations.term = id_term_poly
            splitroiannotations.showTerm = True            
            splitroiannotations.job = job.id
            splitroiannotations.user = user
            splitroiannotations.showWKT = True
            splitroiannotations.fetch()

            for i, roi in enumerate(splitroiannotations):
                # print(roi.area)
                # if roi.area > 800:
                roi_geometry = wkt.loads(roi.location)

                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                roi.dump(dest_pattern=roi_png_filename)                    
                roi.delete() #delete splitpoly annotation to avoid double annotations, we want to keep only the classified annotations
                
                # check white patches
                J = cv2.imread(roi_png_filename,0) #read image and convert to grayscale    
                [r, c]=J.shape
                totalpixels=r*c
                th_remove=parameters.th_remove #more than 10% of patch having white/light area
                white_patch_th = math.floor(totalpixels * th_remove)
                hist,bins = np.histogram(J.ravel(),16,[0,256])
                
                # print(hist[15])
                # print(white_patch_th)
                
                if hist[15] < white_patch_th:

                    if r > 256 or c > 256:
                        JC = cv2.imread(roi_png_filename) #read image in RGB
                        scale_percent = .5 #.5 is 50%
                        width = int(c * scale_percent)
                        height = int(r * scale_percent)
                        dim = (width, height)                        
                        JC2 = cv2.resize(JC, dim, interpolation = cv2.INTER_AREA)

                    else:
                        JC2 = cv2.imread(roi_png_filename) #read image in RGB
                    
                    #Start densenet classification
                    im = cv2.cvtColor(JC2,cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im,(224,224))
                    im = im.reshape(-1,224,224,3)
                    output = np.zeros((0,checkpoint["num_classes"]))
                    arr_out_gpu = torch.from_numpy(im.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)
                    output_batch = model(arr_out_gpu)
                    output_batch = output_batch.detach().cpu().numpy()                
                    output = np.append(output,output_batch,axis=0)
                    pred_labels = np.argmax(output, axis=1)
                    # pred_labels=[pred_labels]

                    if pred_labels[0]==0:
                        # print("Class 0: Normal")
                        id_terms=parameters.normal_term
                        pred_c0=pred_c0+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==1:
                        # print("Class 1: LHP")
                        id_terms=parameters.lhp_term
                        pred_c1=pred_c1+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==2:
                        # print("Class 2: NPI")
                        id_terms=parameters.npi_term
                        pred_c2=pred_c2+1
                        # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                    elif pred_labels[0]==3:
                        # print("Class 3: NPC")
                        id_terms=parameters.npc_term
                        pred_c3=pred_c3+1
                
                    if id_terms!=0:
                        cytomine_annotations = AnnotationCollection()
                        annotation=roi_geometry                    
                        cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                            id_image=id_image,#conn.parameters.cytomine_id_image,
                                                            id_project=project.id,
                                                            id_terms=[id_terms]))
                        print(".",end = '',flush=True)

                        #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                        ca = cytomine_annotations.save()
        
                                           
    finally:
        job.update(progress=100, statusComment="Run complete.")
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")
        
if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

                  






