# Ski_jumping_data_center
Hello. As a big ski jumping fan, I would like to invite everybody to something like a project called "Ski Jumping Data Center". Primary goal is as below:

"Collect as many data about ski-jumping as possible and create as many useful insights based on them as possible"

In the mid-September last year (12.09.20) I thought "Hmm, I don't know any statistical analyses of ski jumping". In fact, the only easily found public data analysis about SJ I know is 
https://rstudio-pubs-static.s3.amazonaws.com/153728_02db88490f314b8db409a2ce25551b82.html

Question is: why? This discipline is in fact overloaded with data, but almost nobody took this topic seriously. Therefore I decided to start collecting data and analyzing them. However, the amount of work needed to capture various data (i.e. jumps and results of competitions) was so big and there is so many ways to use these informations, that make it public was obvious. In fact, I have a plan to expand my database to be as big as possible, but it requires more time and (I wish) more help.

My script now (06.01.21) is able to:
- scrap information from FIS official website and automatically download PDFs (alert: FIS allows to use these methods, see https://www.fis-ski.com/robots.txt), it is done in untitled6.py file
- gather information about competitions in a systematic way - by the script we can export a database with basic facts about every PDF with results (untitled0.py) and build a foundation to parsing results,
- parse results of every World Cup/Grand Prix competition from 2009/10 season to now, including these without gate/wind compensations, along with World Championships and Ski Flying World Championships (tested with success back to 2009/10 season), coded in untitled0.py
- parse results of every Continental Cup/FIS Cup competition,
- parse results of training/trial rounds from all described above types of competitions (with minor exceptions)
- create a database containing any jump from results parsed above (untitled8.py) and create rating of each jumper. It's based on Elo rating system (see https://en.wikipedia.org/wiki/Elo_rating_system for details) and allows to compare the potential of athletes. 

Next steps are as below:
- complete the documentation of function appearing in the modules,
- optimize code (make the style better, clean redundant sections etc.)
- make some insights from the collected data, at least one.
My scripts have several limitations and I take no responsibility for errors appearing in them.

