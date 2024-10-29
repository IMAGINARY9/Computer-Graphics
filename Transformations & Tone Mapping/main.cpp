/**
@file main.cpp
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "../glm/glm.hpp"
#include "../glm/gtx/transform.hpp"

#include "Image.h"
#include "Material.h"

# define M_PI           3.14159265358979323846  /* pi */

using namespace std;

/**
 Class representing a single ray.
 */
class Ray{
public:
    glm::vec3 origin; ///< Origin of the ray
    glm::vec3 direction; ///< Direction of the ray
	/**
	 Contructor of the ray
	 @param origin Origin of the ray
	 @param direction Direction of the ray
	 */
    Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction){
    }
};


class Object;

/**
 Structure representing the even of hitting an object
 */
struct Hit{
    bool hit; ///< Boolean indicating whether there was or there was no intersection with an object
    glm::vec3 normal; ///< Normal vector of the intersected object at the intersection point
    glm::vec3 intersection; ///< Point of Intersection
    float distance; ///< Distance from the origin of the ray to the intersection point
    Object *object; ///< A pointer to the intersected object
};

/**
 General class for the object
 */
class Object{
	
protected:
	glm::mat4 transformationMatrix; ///< Matrix representing the transformation from the local to the global coordinate system
	glm::mat4 inverseTransformationMatrix; ///< Matrix representing the transformation from the global to the local coordinate system
	glm::mat4 normalMatrix; ///< Matrix for transforming normal vectors from the local to the global coordinate system
	
public:
	glm::vec3 color; ///< Color of the object
	Material material; ///< Structure describing the material of the object
	/** A function computing an intersection, which returns the structure Hit */
    virtual Hit intersect(Ray ray) = 0;

	/** Function that returns the material struct of the object*/
	Material getMaterial(){
		return material;
	}
	/** Function that set the material
	 @param material A structure describing the material of the object
	*/
	void setMaterial(Material material){
		this->material = material;
	}
	/** Functions for setting up all the transformation matrices
	@param matrix The matrix representing the transformation of the object in the global coordinates */
	void setTransformation(glm::mat4 matrix){
			
		transformationMatrix = matrix;
			
		/* ----- Exercise 2 ---------
		Set the two remaining matrices

		*/

		inverseTransformationMatrix = glm::inverse(transformationMatrix);
		normalMatrix = glm::transpose(inverseTransformationMatrix);

	}
};

/**
 Implementation of the class Object for sphere shape.
 */
class Sphere : public Object{
private:
    float radius; ///< Radius of the sphere
    glm::vec3 center; ///< Center of the sphere

public:
	/**
	 The constructor of the sphere
	 @param radius Radius of the sphere
	 @param center Center of the sphere
	 @param color Color of the sphere
	 */
    Sphere(float radius, glm::vec3 center, glm::vec3 color) : radius(radius), center(center){
		this->color = color;
    }
	Sphere(float radius, glm::vec3 center, Material material) : radius(radius), center(center){
		this->material = material;
	}
	/** Implementation of the intersection function*/
    Hit intersect(Ray ray){

        glm::vec3 c = center - ray.origin;

        float cdotc = glm::dot(c,c);
        float cdotd = glm::dot(c, ray.direction);

        Hit hit;

        float D = 0;
		if (cdotc > cdotd*cdotd){
			D =  sqrt(cdotc - cdotd*cdotd);
		}
        if(D<=radius){
            hit.hit = true;
            float t1 = cdotd - sqrt(radius*radius - D*D);
            float t2 = cdotd + sqrt(radius*radius - D*D);

            float t = t1;
            if(t<0) t = t2;
            if(t<0){
                hit.hit = false;
                return hit;
            }

			hit.intersection = ray.origin + t * ray.direction;
			hit.normal = glm::normalize(hit.intersection - center);
			hit.distance = glm::distance(ray.origin, hit.intersection);
			hit.object = this;
        }
		else{
            hit.hit = false;
		}
		return hit;
    }
};

class Plane : public Object{

private:
	glm::vec3 normal;
	glm::vec3 point;

public:
	Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal){
	}
	Plane(glm::vec3 point, glm::vec3 normal, Material material) : point(point), normal(normal){
		this->material = material;
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;
		
		//Plane-ray intersection

		float denom = glm::dot(normal, ray.direction);

		if (denom != 0)
		{
			glm::vec3 p = point - ray.origin;

			float t = glm::dot(p, normal) / denom;

			if (t > 0)
			{
				hit.hit = true;
				hit.distance = t;
				hit.normal = normal;
				hit.intersection = ray.origin + t * ray.direction;
				hit.object = this;
			}
		}

		return hit;
	}
};

class Cone : public Object{
private:
	Plane *plane;
	
public:
	Cone(Material material){
		this->material = material;
		plane = new Plane(glm::vec3(0,1,0), glm::vec3(0.0,1,0));
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;
		
	
		/*  ---- Exercise 2 -----
		
		 Implement the ray-cone intersection. Before intersecting the ray with the cone,
		 make sure that you transform the ray into the local coordinate system.
		 Remember about normalizing all the directions after transformations.
		 
		*/
	
		/* If the intersection is found, you have to set all the critical fields in the Hit strucutre
		 Remember that the final information about intersection point, normal vector and distance have to be given
		 in the global coordinate system.
		
		 */

		glm::vec3 dir = glm::normalize(inverseTransformationMatrix * glm::vec4(ray.direction, 0.0f));
		glm::vec3 orig = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0f);

		auto a = dir.x * dir.x + dir.z * dir.z - dir.y * dir.y;
		auto b = 2 * (dir.x * orig.x + dir.z * orig.z - dir.y * orig.y);
		auto c = orig.x * orig.x + orig.z * orig.z - orig.y * orig.y;

		float delta = b * b - 4 * a * c;

		if (delta > 0)
		{

			float t1 = (-b - sqrt(delta)) / (2 * a);
			float t2 = (-b + sqrt(delta)) / (2 * a);

			float t = t1;
			hit.intersection = orig + t * dir;
			if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
			{
				t = t2;
				hit.intersection = orig + t * dir;
				if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
				{
					return hit;
				}
			}

			hit.normal = glm::normalize(glm::vec3(hit.intersection.x, -hit.intersection.y, hit.intersection.z));

			Ray local_ray(orig, dir);
			Hit hit_plane = plane->intersect(local_ray);
			if (hit_plane.hit && hit_plane.distance < t
				&& length(hit_plane.intersection - glm::vec3(0, 1, 0)) <= 1.0f)
			{
				hit.intersection = hit_plane.intersection;
				hit.normal = hit_plane.normal;
			}

			hit.hit = true;
			hit.object = this;
			hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0f);
			hit.normal = glm::vec3(normalMatrix * glm::vec4(hit.normal, 0.0f));
			hit.normal = glm::normalize(hit.normal);
			hit.distance = glm::distance(hit.intersection, ray.origin);

		}
		
		return hit;
	}
};

/**
 Light class
 */
class Light{
public:
	glm::vec3 position; ///< Position of the light source
	glm::vec3 color; ///< Color/intentisty of the light source
	Light(glm::vec3 position): position(position){
		color = glm::vec3(1.0);
	}
	Light(glm::vec3 position, glm::vec3 color): position(position), color(color){
	}
};

vector<Light *> lights; ///< A list of lights in the scene
glm::vec3 ambient_light(0.001,0.001,0.001);
vector<Object *> objects; ///< A list of all objects in the scene


/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computer
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 view_direction, Material material){

	glm::vec3 color(0.0);
	for(int light_num = 0; light_num < lights.size(); light_num++){

		glm::vec3 light_direction = glm::normalize(lights[light_num]->position - point);
		glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);

		float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
		float VdotR = glm::clamp(glm::dot(view_direction, reflected_direction), 0.0f, 1.0f);

		glm::vec3 diffuse_color = material.diffuse;
		glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);
		glm::vec3 specular = material.specular * glm::vec3(pow(VdotR, material.shininess));
		
		/*  ---- Exercise 3-----
		
		 Include light attenuation due to the distance to the light source.
		 
		*/

		float ldist = max(glm::distance(point, lights[light_num]->position), 0.1f);
		
		color += lights[light_num]->color * (diffuse + specular) / (ldist * ldist);
		
	
	}
	color += ambient_light * material.ambient;
	color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
	return color;
}

/**
 Functions that computes a color along the ray
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray){

	Hit closest_hit;

	closest_hit.hit = false;
	closest_hit.distance = INFINITY;

	for(int k = 0; k<objects.size(); k++){
		Hit hit = objects[k]->intersect(ray);
		if(hit.hit == true && hit.distance < closest_hit.distance)
			closest_hit = hit;
	}

	glm::vec3 color(0.0);

	if(closest_hit.hit){
		color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->getMaterial());
	}else{
		color = glm::vec3(0.0, 0.0, 0.0);
	}
	return color;
}
/**
 Function defining the scene
 */
void sceneDefinition (){

	/*  ---- All Exercises -----
	
	 Modify the scene definition according to the exercises
	 
	*/

	Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = glm::vec3(0.3f, 1.0f, 0.3f);

	Material red_specular;
	red_specular.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = glm::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;
	
	
	objects.push_back(new Sphere(1.0, glm::vec3(1,-2,8), blue_specular));
	objects.push_back(new Sphere(0.5, glm::vec3(-1,-2.5,6), red_specular));

	//

	Material yellow_specular;
	yellow_specular.ambient = glm::vec3(0.2f);
	yellow_specular.diffuse = glm::vec3(0.5f, 0.5f, 0.0f);
	yellow_specular.specular = glm::vec3(0.6f);
	yellow_specular.shininess = 100.0;

	Cone* yellow_cone = new Cone(yellow_specular);
	glm::mat4 translationMatrix = glm::translate(glm::vec3(5, 9, 14));
	glm::mat4 scalingMatrix = glm::scale(glm::vec3(3.0f, 12.0f, 3.0f));
	glm::mat4 rotationMatrix = glm::rotate(glm::radians(180.0f), glm::vec3(1, 0, 0));
	yellow_cone->setTransformation(translationMatrix * scalingMatrix * rotationMatrix);

	Cone* green_cone = new Cone(green_diffuse);
	translationMatrix = glm::translate(glm::vec3(6, -3, 7));
	scalingMatrix = glm::scale(glm::vec3(1.0f, 3.0f, 1.0f));
	rotationMatrix = glm::rotate(glm::atan(3.0f), glm::vec3(0, 0, 1));
	green_cone->setTransformation(translationMatrix * rotationMatrix * scalingMatrix);
	
	objects.push_back(yellow_cone);
	objects.push_back(green_cone);

	//

	Material purple_wall;
	purple_wall.ambient = glm::vec3(0.05f, 0.05f, 0.07f);
	purple_wall.diffuse = glm::vec3(0.5f, 0.5f, 0.7f);
	purple_wall.specular = glm::vec3(0.9);
	purple_wall.shininess = 10.0;

	Material pink_wall;
	pink_wall.ambient = glm::vec3(0.08f, 0.05f, 0.05f);
	pink_wall.diffuse = glm::vec3(0.8f, 0.5f, 0.5f);
	pink_wall.specular = glm::vec3(0.3);
	pink_wall.shininess = 100.0;
	
	Material white_diffuse;
	white_diffuse.ambient = glm::vec3(0.06f, 0.06f, 0.06f);
	white_diffuse.diffuse = glm::vec3(0.6f, 0.6f, 0.6f);
	white_diffuse.shininess = 10.0;

	objects.push_back(new Plane(glm::vec3(15, 0, 0), glm::vec3(-1, 0, 0), purple_wall));
	objects.push_back(new Plane(glm::vec3(-15, 0, 0), glm::vec3(1, 0, 0), pink_wall));
	
	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0, 1, 0), white_diffuse));
	objects.push_back(new Plane(glm::vec3(0, 27, 0), glm::vec3(0, -1, 0), white_diffuse));

	objects.push_back(new Plane(glm::vec3(0, 0, -0.01), glm::vec3(0, 0, 1), green_diffuse));
	objects.push_back(new Plane(glm::vec3(0, 0, 30), glm::vec3(0, 0, -1), green_diffuse));
	
	//

	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));
}
glm::vec3 toneMapping(glm::vec3 intensity){

	/*  ---- Exercise 3-----
	
	 Implement a tonemapping strategy and gamma correction for a correct display.
	 
	*/

	float gamma = 2.25f;
	float alpha = 10.0f;
	return alpha * glm::pow(intensity, glm::vec3(1 / gamma));
	
	return intensity;
}
int main(int argc, const char * argv[]) {

    clock_t t = clock(); // variable for keeping the time of the rendering

    int width = 1024; //width of the image
    int height = 768; // height of the image
    float fov = 90; // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width,height); // Create an image where we will store the result

    float s = 2*tan(0.5*fov/180*M_PI)/width;
    float X = -s * width / 2;
    float Y = s * height / 2;

    for(int i = 0; i < width ; i++)
        for(int j = 0; j < height ; j++){

			float dx = X + i*s + s/2;
            float dy = Y - j*s - s/2;
            float dz = 1;

			glm::vec3 origin(0, 0, 0);
            glm::vec3 direction(dx, dy, dz);
            direction = glm::normalize(direction);

            Ray ray(origin, direction);

			image.setPixel(i, j, glm::clamp(toneMapping(trace_ray(ray)), glm::vec3(0.0), glm::vec3(1.0)));

        }

    t = clock() - t;
    cout<<"It took " << ((float)t)/CLOCKS_PER_SEC<< " seconds to render the image."<< endl;
    cout<<"I could render at "<< (float)CLOCKS_PER_SEC/((float)t) << " frames per second."<<endl;

	// Writing the final results of the rendering
	if (argc == 2){
		image.writeImage(argv[1]);
	}else{
		image.writeImage("./result.ppm");
	}

	
    return 0;
}
