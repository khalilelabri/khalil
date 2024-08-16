import camelot
import fitz
import re
import numpy as np
import cv2
import os
import json

def extract_tables_helper(pdf_document,pdf_path,page_number):
    extracted_tables = camelot.read_pdf(pdf_path, line_scale=40, pages=str(page_number+1),flavor="lattice",copy_text=['v','h'])   
    tables_list=[]
    matches=[]
    excluded_area=list()
    page = pdf_document.load_page(page_number)
    for table in extracted_tables:
        x1,y1,x2,y2=table._bbox
        rect=fitz.Rect(0,page.rect.height-y2-20,page.rect.width,page.rect.height-y2)
        text_above = page.get_textbox(rect)
        if re.search(r'Table \d+\..*',text_above):
            tables_list.append(table.df)
            match = re.findall(r'Table \d+\..*', text_above)[-1]
            matches.append(match)
            excluded_area.append((x1,y1,x2,y2+20))
        if re.sub(r'\s+', '', text_above).isdigit():
            excluded_area.append((x1,y1,x2,y2+20))
    titles = [title for match in matches for title in (match if isinstance(match, list) else [match])]
    tables = [title for match in tables_list for title in (match if isinstance(match, list) else [match])]
    return titles,tables,excluded_area

def extract_tables(pdf_document,pdf_path,page_number,_index,output_path="../Ressources/Tables"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    titles,tables,excluded_area=extract_tables_helper(pdf_document,pdf_path,page_number)
    assert len(tables)==len(titles)
    try:

        for title,table in zip(titles,tables):
            table.to_csv("{}/{}.csv".format(output_path,re.sub(r'[\\/:*?"<>|]',' ',title)+"_{}".format(_index)),header=False,index=False)
            _index+=1
    except:
        pass
    return excluded_area

def extract_figures(pdf_document,page_number,output_path="../Ressources/Images"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    page = pdf_document[page_number]
    width=page.rect.width
    height=page.rect.height
    excluded_area=list()
    zoom = 5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n - pix.alpha < 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    image_list=[]
    image_names=[]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        rect=fitz.Rect(0,(y//5)-50,width,y//5)
        text_above = page.get_textbox(rect)
        if re.search(r'Figure \d+\..*',text_above):
            cv2.rectangle(img, (x, y), (x+w, y+h), (36, 255, 12), 3)
            cropped=img[y:y+h,x:x+w]
            image_list.append(cropped)
            matches =re.findall(r'Figure \d+\..*', text_above)[-1]
            image_names.append(matches)
            excluded_area.append((x//5,height-(y//5)-(h//5),(x//5)+(w//5),height-(y//5)+15))
    assert len(image_list)==len(image_names)
    names=[title for match in image_names for title in (match if isinstance(match, list) else [match])]
    try:

        for name,image in zip(names,image_list):
            cv2.imwrite("{}/{}.png".format(output_path,re.sub(r'[\\/:*?"<>|]',' ',name)),image)
    except:
        pass

    return excluded_area

def get_area(page,excluded_area):
    remaining_area=[(0,110,page.rect.width,page.rect.height-75)]
    sorted_list = sorted(excluded_area, key=lambda x: x[1])
    for area in sorted_list:
        x1,y1,x2,y2 = remaining_area.pop()
        below=(x1,y1,x2,area[1])
        above=(x1,area[-1],x2,y2)
        if area[1]-y1>5:
            remaining_area.append(below)
        if y2-area[-1]>5:
            remaining_area.append(above)
    return remaining_area[::-1]


def extract_text(pdf_document,page_number,excluded_area):
    page=pdf_document.load_page(page_number)
    rect=fitz.Rect(0,0,page.rect.width,75)
    header = str(page.get_textbox(rect)).lower()
    if ("content" in header) or ("list of tables" in header) or ("list of figures" in header) or ("index" in header):
        return ""
    text=""
    reamining_area=get_area(page,excluded_area)
    for area in reamining_area:
        rect=fitz.Rect(area[0],page.rect.height-area[3],area[2],page.rect.height-area[1])
        text+= page.get_text("text",clip=rect)
    return text


def clean_text(text):
    ch=re.sub(r'\s+',' ',text)
    ch=re.sub(r'\n+',' ',text)
    return ch.strip()

def write_json(dictionary,filename,output_path="../Ressources/Json"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open("{}/{}.json".format(output_path,filename),'w') as file : 
        json.dump(dictionary,file)


def split_by_title(raw_text,titles_list):
    text=raw_text.replace('  ',' ').lower().strip()
    title_pos_list=list()
    title_content_dict=dict()
    for title in titles_list:
        pos=text.find(title.lower().strip())
        if pos !=-1:
            title_pos_list.append([title.lower().strip(),pos])
        else:
            titles_list.remove(title)
    for i in range(len(title_pos_list)-1) :
        key=title_pos_list[i][0]
        start=title_pos_list[i][1]+len(title_pos_list[i][0])
        end=title_pos_list[i+1][1]
        value=text[start:end]
        title_content_dict[key]=value
    if title_pos_list[0][1]!=0:
        title_content_dict["untitled"]= text[0:title_pos_list[0][1]]

    return title_content_dict


def pdf_extractor(pdf_path,output_path="../Ressources/"):
    filename=pdf_path.split("/")[-1].split('.')[0]
    pdf_document = fitz.open(pdf_path)
    text=list()
    contents=list()
    i=1
    for page_number in range(len(pdf_document)):
        page=pdf_document.load_page(page_number)
        rect=fitz.Rect((0,0,page.rect.width,75))
        header = str(page.get_textbox(rect)).lower()
        if "content" in header:
            texts=page.get_text("text",clip=fitz.Rect((0,75,page.rect.width,page.rect.height-110)))
            texts=texts.split("\n")
            texts=[i for i in texts if ((len(i.strip())>0) and ("contents" != i.strip().lower()))]
            start=0
            for i in range(len(texts)):
                l=texts[i].split('.')
                if len(l)>3:
                    title=l[0]
                    full_title=' '.join(texts[start:i])+' '+title.strip()
                    contents.append(full_title)
                    start=i+1
        contents=[re.sub(r'\s+',' ',i.strip()) for i in contents if not (i.strip().isdigit())]
        excluded_area=list()
        path=output_path+filename+"/"
        excluded_figure_area=extract_figures(pdf_document,page_number,output_path=path+"Images")
        excluded_table_area=extract_tables(pdf_document,pdf_path,page_number,_index=i,output_path=path+"Tables")
        i+=1
        excluded_area.extend(excluded_figure_area)
        excluded_area.extend(excluded_table_area)
        extracted_text=extract_text(pdf_document,page_number,excluded_area)
        if len(extracted_text)>1:
            text.append(clean_text(extracted_text))

    raw_text='\n'.join(text)
    print(contents)
    dictionary=split_by_title(raw_text,contents)
    write_json(dictionary,filename,output_path=path+"Json")
        
    pdf_document.close()
    return dictionary
