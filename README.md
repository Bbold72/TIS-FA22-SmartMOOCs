# Intelligent Learning Platforms: Segmenting Lecture Videos Based on Topic Change

| **Name**              | **User ID**     | **Role**       |
|-----------------------|-----------------|----------------|
|     Brian Reinbold    |     brianjr3    |     Captain    |


This project explores better ways to segment lectures based on topic transitions for the SmartMOOCs platform. See `proposal.pdf` for details.

# Documentation

## Setup
Use our pre-saved conda environment

```
conda env create --name tis-project --file=environment.yml
conda activate tis-project
```

or try to install from the requirement.txt

```
pip3 install -r requirements.txt
```

then install project as a package

```
pip install -e .
```
For details on how to use Jupyter notebooks locally, see the [documentation](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)

Also, [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) natively supports running Jupyter notebooks, which may be easier.



## Running Project
I developed the project using [https://learn.microsoft.com/en-us/windows/wsl/install](Windows Subsystem for Linux) on Ubuntu 20.04. Any Unix based OS should not have an issue running project. Windows should be fine too but it may be difficult to use the Makefile, however, there are instructions to run each script individually if necessary.    

Run the Jupyter notebook `./notebooks/demo.ipynb` to replicate key results of project for one sample lesson, "Week 4 Lesson 1: Probabilistic Retrieval Model: Basic Idea"     

If you have issues running Juputer notebooks locally, you can run the demo as a python script.
```
python notebooks/demo.py
```


The projects uses a Makefile to keep track of dependendicies and make it easier to replicate the project.  

To run project on all transcript files:
```
make run
```
This will output a file `final_results.csv` in the data folder.

Alternatively, can run the commands individually:

1. Process raw transcript files
```
make transcripts
```
2. Create corpus out of transcript segments to calculate term-document frequency matrix and time series of similarity
```
make corpus
```
3. Estimate and evaluate time series breakpoints
```
make breakpoints
```

If there are issues with make, you can run the python scripts directly:
1. Process raw transcript files
```
python src/data/process_transcripts.py
```
2. Create corpus out of transcript segments to calculate term-document frequency matrix and time series of similarity
```
python src/data/make_corpus.py
```
3. Estimate and evaluate time series breakpoints
```
python src/models/breakpoints.py
```


### Using Docker
If you are familiar with Docker, you can also use it to run the project although it is not necessary.

Build image
```
docker build -t tis .
```

Running container and automatically run demo.py file
```
docker run tis
```

Attaching terminal to container. This allows you to run files individually and run using make.
```
docker run -it tis sh
make run
```

## Directory Structure
Project follows the structure outlined in [CookieCutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).    
<pre>
📦data             - stores data files   
 ┣ 📂intermediate  - processed data is stored here   
 ┗ 📂raw           - raw transcript files organized by week and lesson   
   ┗ 📂cs-410   
📦notebooks        - Jupyter notebooks of data exploration and analysis   
 ┣ 📜demo.ipynb    - demo of key project findings   
 ┗ 📜vocab.ipynb   - data exploration of vocabulary used in all TIS    lectures   
 📦src             - scripts for project    
 ┣ 📂data          - scripts to process data   
 ┣ 📂models        - scripts model and evaluate time series breakpoints   
 ┗ 📜utils.py      - module of helper functions    
</pre>

## Downloading the Transcripts
Although not necessary since the raw files are saved to the repo, here were the steps to download the raw transcript file from Coursera.   
To download all video transcripts, pull the [coursera-dl](https://github.com/coursera-dl/coursera-dl) repo and use the `coursera-dl` script. The command below will download all raw transcript "txt" file and annotated transcript "srt" files:

```
./coursera-dl -ca {CAUTH} -f "srt txt" --subtitle-language en cs-410
```
- -ca: CAUTH token   
    1. First login to coursera.org from your web browser
    1. In Chrome, go to web browser settings
    1. Advanced
    1. Privacy and Security
    1. Site Settings
    1. Cookies and Site Data
    1. See all cookies and site data
    1. Find coursera.org, click into it and check for CAUTH
    1. Copy this value into `-ca` flag. Note that this value can be very large
- -f: filter course contents by file extension
- --subtitle-language: language of subtitles to downloald. Use `en` for english.    


If you get an error that starts like this:
```
HTTPError 404 Client Error: Not Found for url: https://api.coursera.org/api/onDemandCourseMaterials.v1/
```
The API endpoints are outdated in `coursera-dl` and you'll need to update them. Change lines 318-322 in `coursera-dl/api.py` to:
```python
        dom = get_page(session, OPENCOURSE_ONDEMAND_COURSE_MATERIALS_V2,
                       json=True,
                       class_name=course_name)
        return OnDemandCourseMaterialItemsV1(
            dom['linked']['onDemandCourseMaterialItems.v2'])
```
