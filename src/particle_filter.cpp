/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// random engine used across various methods
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Create normal distributions for x, y and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100;

  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;

    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Create normal distributions for x, y and theta.
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {
    // Add measurements to each particle
    if (fabs(yaw_rate) < 1e-6)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // particles[i].theta += 0;
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // add random Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++)
  {
    LandmarkObs o = observations[i];

    // initialize minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs p = predicted[j];

      // get distance between current/predicted landmarks
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      // set the nearest the predicted measurement
      if (cur_dist < min_dist)
      {
        min_dist = cur_dist;
        observations[i].id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
  for (int i = 0; i < num_particles; i++)
  {
    // particle coordinates
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // landmark locations within sensor range
    vector<LandmarkObs> filtered_landmarks;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      // landmark coordinates
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // landmarks within sensor range
      if (dist(landmark_x, landmark_y, particle_x, particle_y) <= sensor_range)
      {
        filtered_landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // transfor observations to map coordinates
    vector<LandmarkObs> observations_map;
    for (int j = 0; j < observations.size(); j++)
    {
      double t_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particle_x;
      double t_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particle_y;
      observations_map.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    // find predicted measurement closest to each observed measurement
    dataAssociation(filtered_landmarks, observations_map);

    // new weight
    particles[i].weight = 1;

    for (int j = 0; j < observations_map.size(); j++)
    {

      double observation_x, observation_y, prediction_x, prediction_y;
      observation_x = observations_map[j].x;
      observation_y = observations_map[j].y;

      // prediction coordinates
      for (int k = 0; k < filtered_landmarks.size(); k++)
      {
        if (filtered_landmarks[k].id == observations_map[j].id)
        {
          prediction_x = filtered_landmarks[k].x;
          prediction_y = filtered_landmarks[k].y;
          break;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
      double exponent = pow(prediction_x - observation_x, 2) / (2 * pow(sig_x, 2)) + (pow(prediction_y - observation_y, 2) / (2 * pow(sig_y, 2)));

      // product of this obersvation weight with total observations weight
      particles[i].weight *= gauss_norm * exp(-exponent);
    }
  }
}

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // new particles vector
  vector<Particle> new_particles;

  // current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }

  // uniform random distribution [0, 1]
  std::uniform_real_distribution<> unirealdist(0, 1);

  int index = int(unirealdist(gen) * num_particles);
  double mw = *max_element(weights.begin(), weights.end());
  double beta = 0.0;

  for (int i = 0; i < num_particles; i++)
  {
    beta += mw * unirealdist(gen) * 2.0;

    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
