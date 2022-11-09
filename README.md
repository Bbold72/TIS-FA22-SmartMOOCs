# Intelligent Learning Platforms: Segmenting Lecture Videos Based on Topic Change

| **Name**              | **User ID**     | **Role**       |
|-----------------------|-----------------|----------------|
|     Brian Reinbold    |     brianjr3    |     Captain    |


This project explores better ways to segment lectures based on topic transitions for the SmartMOOCs platform. See `proposal.pdf` for details.


## Downloading the Transcripts
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






