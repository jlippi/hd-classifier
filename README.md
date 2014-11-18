####What is this product?

This is a helpdesk ticket classification product. It is set up to run with django's tickets (available here: https://code.djangoproject.com/query) and tickets from the github project saltstack/salt.  It can be easily configured to run on any ticketing system that works with github, jira, or rpc's APIs.

####Why do I care?
I'm glad you're asking this question. I used to work at a company with ~1500 employees. In a company that size, you need a helpdesk to do a lot of things, and someone has to triage where the tickets go.

I know that if turnaround time could be improved by even a minute per ticket by reducing the amount of work done to triage, this could save the company thousands of hours of productivity a year.

Auto categorization could also help detect important tickets (ie "the building is on fire") quickly, and future algorithms could recognize patterns in incoming tickets to help detect production problems.

####So what exactly does it do?
This product automatically classifies all tickets from the django ticket tracker and classifies them based on assigned labels of other tickets.

The algorithm it uses is very simple: TFIDF with multinomial logistic regression and Latent Dirichlet Allocation for a very marginal improvement.

####How to use it:

To just get this up and running, do the following:
- Clone the archive
- Make sure you have mongodb server installed and running.
- $ python run_django.py (this may take a very long time, up to several hours to do all the scraping and fit the LDA.  Fortunately the big chunk of scraping only needs to happen once and the model only needs to be trained infrequently)
- $ python app.py
- Navigate to http://0.0.0.0:8142/

By default, the most recent 100 tickets are shown. The tickets furthest to the right are the newest.

The tickets closest to the top are the ones that are most likely to be the label that is being selected

To change the label that's selected, just click one of the buttons below the visualization.  This will change the sort order of the tickets.
