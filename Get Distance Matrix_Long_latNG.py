
"""
https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=6.438722,7.486025&destinations=6.031176%2C7.576712%7C6.0828972C7.475118%7C6.51472%2C7.512057%7C6.388048%2C7.289398%7C6.425608%2C7.492846&key=AIzaSyCjr6cPndDG3AmFgWSlh3CnCVd7SqjmUGc
https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=6.438722,7.486025&destinations=6.031176%2C7.576712%7C6.0828972C7.475118%7C6.51472%2C7.512057%7C6.388048%2C7.289398%7C6.425608%2C7.492846%7C6.388048%2C7.289398
%7C7.410309%7C7.410309%7C7.441261%7C7.403779%7C7.702478%7C7.649427%7C6.388048&key=AIzaSyCjr6cPndDG3AmFgWSlh3CnCVd7SqjmUGc

"""

#function builds rows of the distance matrix, using the response returned by the send_request function.
def build_distance_matrix(response):
  distance_matrix = []
  for row in response['rows']:
    row_list = [row['elements'][j]['distance']['value'] for j in range(len(row['elements']))]
    distance_matrix.append(row_list)
  return distance_matrix