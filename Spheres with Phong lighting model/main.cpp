/**
@file main.cpp
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#include "Image.h"
#include "Material.h"

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
	/** Implementation of the Ray-sphere intersection function*/
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;

		glm::vec3 c = this->center - ray.origin;	

		float a = glm::dot(c, ray.direction);	

		float c2 = glm::dot(c, c);	
		float a2 = a * a;

		float D = glm::sqrt(c2 - a2);

		float D2 = c2 - a2;
		float r2 = radius * radius;

		if (D > radius)
			return hit;

		float b = sqrt(r2 - D2);

		float t1 = a - b;
		float t2 = a + b;

		float t = (t1 > 0) ? t1 : t2;

		if (t < 0)	// Both intersections are behind the ray's origin
			return hit;

		hit.hit = true;
		hit.distance = t;

		hit.intersection = ray.origin + t * ray.direction;
		hit.normal = glm::normalize(hit.intersection - center);

		hit.object = this;
		
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
glm::vec3 ambient_light(0.1,0.1,0.1);
vector<Object *> objects; ///< A list of all objects in the scene


/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computer
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 view_direction, Material material){

	glm::vec3 color(0.0);

	for (Light* l : lights)
	{
		glm::vec3 source = glm::normalize(l->position - point);

		auto diffuse = max(glm::dot(source, normal), 0.0f) * material.diffuse;

		auto reflection = glm::reflect(-source, normal);

		auto specular = glm::pow(max(glm::dot(view_direction, reflection), 0.0f), material.shininess) * material.specular;

		color += l->color * (diffuse + specular);
	}

	color += ambient_light * material.ambient;
	
	// The final color has to be clamped so the values do not go beyond 0 and 1.
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

		//color = closest_hit.object->color;

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

	//objects.push_back(new Sphere(1.0, glm::vec3(0,-2,8), glm::vec3(0.6, 0.9, 0.6)));
	
	//objects.push_back(new Sphere(1.0, glm::vec3(1,-2,8), glm::vec3(0.6, 0.6, 0.9)));
	

	 //// Example of adding one sphere:
	 //
	 //Material red_specular;
	 //red_specular.diffuse = glm::vec3(1.0f, 0.3f, 0.3f);
	 //red_specular.ambient = glm::vec3(0.01f, 0.03f, 0.03f);
	 //red_specular.specular = glm::vec3(0.5);
	 //red_specular.shininess = 10.0;
	 //
	 //objects.push_back(new Sphere(0.5, glm::vec3(-1,-2.5,6), red_specular));
	 //
	 //// Example a white light of intensity 0.4 and position in (0,26,5):
	 //
	 //lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(0.4)));
	 

	Material red_specular;
	red_specular.ambient = glm::vec3(0.01f, 0.03f, 0.03f);
	red_specular.diffuse = glm::vec3(1.0f, 0.3f, 0.3f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;
	objects.push_back(new Sphere(0.5, glm::vec3(-1, -2.5, 6), red_specular));

	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.07f, 0.07f, 0.01f);
	blue_specular.diffuse = glm::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;
	objects.push_back(new Sphere(1.0, glm::vec3(1, -2, 8), blue_specular));

	Material green_specular;
	green_specular.ambient = glm::vec3(0.07f, 0.09f, 0.07f);
	green_specular.diffuse = glm::vec3(0.7f, 0.9f, 0.7f);
	green_specular.specular = glm::vec3(0.0);
	green_specular.shininess = 0.0;
	objects.push_back(new Sphere(1.0, glm::vec3(2, -2, 6), green_specular));

	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(0.4)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.4)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));

	// Rainbow balls

	//float red, green, blue;
	//float step = 0.25f;

	//for (red = 0; red <= 1; red += step) {
	//	for (green = 0; green <= 1; green += step) {
	//		for (blue = 0; blue <= 1; blue += step) {
	//			Material grb_specular;
	//			grb_specular.ambient = glm::vec3(red, green, blue);
	//			grb_specular.diffuse = glm::vec3(red, green, blue);
	//			grb_specular.specular = glm::vec3(0.8);
	//			grb_specular.shininess = 100;
	//			objects.push_back(new Sphere(1.0, glm::vec3(red * 10, green * 10, blue * 10 + 10), grb_specular));
	//			//objects.push_back(new Sphere(1.0, glm::vec3(red * 10, green * 10, blue * 10 + 10), glm::vec3(red, green, blue)));
	//		}
	//	}
	//}
}

int main(int argc, const char * argv[]) {

    clock_t t = clock(); // variable for keeping the time of the rendering

    int width = 1024; //width of the image
    int height = 768; // height of the image
    float fov = 90; // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width,height); // Create an image where we will store the result
	 
	// Loop over pixels to form and traverse the rays through the scene

	float scale = (2 * tan(fov * 0.5f)) / width;

	float X = -(width * scale) / 2.0f;
	float Y = (height * scale) / 2.0f;

    for(int i = 0; i < width ; i++)
        for(int j = 0; j < height ; j++){

			// Ray definition for pixel (i,j)
		
			float x = X + i * scale + 0.5f * scale;
			float y = Y - j * scale - 0.5f * scale;

			//Definition of the ray
			//glm::vec3 origin(5, 5, 0); // for rainbow balls
			glm::vec3 origin(0, 0, 0);
			glm::vec3 direction(x, y, 1.0f);               
			direction = glm::normalize(direction);
			
			Ray ray(origin, direction);  // ray traversal
			
			image.setPixel(i, j, trace_ray(ray));

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
