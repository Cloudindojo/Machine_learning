from nltk.sentiment.vader import SentimentIntensityAnalyzer
def review_analyser(text):
	
	#message_text = '''Food taste is not that good '''
	
	sid = SentimentIntensityAnalyzer()
	scores = sid.polarity_scores(text)
	te = text
	li = []
	for key in sorted(scores):
	    p = scores[key]
	    q = int(p*100)
	    li.append(q)

	print (li[0])


	def star_cal(val):
		if int(val) in range(1, 21):
			star = 1
		elif int(val) in range(21, 41):
		    star = 2
		elif int(val) in range(41, 61):
		    star = 3
		elif int(val) in range(61, 81):
		    star = 4
		elif int(val) in range(81, 101):
		    star = 5
		else:
		    star = 0
		return(star)

	val = li[0]
	
	
	if val<0:
		term = "negative"
		val = val+100
		star = star_cal(val)
		#print(star)
	elif val>0:
		term = "positive"
		star = star_cal(val)
	else :
		star = 0
		term = "neutral"
		#print(star)
	

	return val,star,term



