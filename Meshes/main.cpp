/**
@file main.cpp
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
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

		inverseTransformationMatrix = glm::inverse(matrix);
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
		
        float DdotN = glm::dot(ray.direction, normal);
        if(DdotN < 0){
            
            float PdotN = glm::dot (point-ray.origin, normal);
            float t = PdotN/DdotN;
            
            if(t > 0){
                hit.hit = true;
                hit.normal = normal;
                hit.distance = t;
                hit.object = this;
                hit.intersection = t * ray.direction + ray.origin;
            }
        }
		
		return hit;
	}
};

class Triangle : public Object {
private:
	glm::vec3 p0, p1, p2;
	glm::vec3 n0, n1, n2;
	bool smooth;

public:
	Triangle(glm::vec3 p0, glm::vec3  p1, glm::vec3 p2) : p0(p0), p1(p1), p2(p2), smooth(false) {}
	Triangle(glm::vec3 p0, glm::vec3  p1, glm::vec3 p2, glm::vec3 color) : p0(p0), p1(p1), p2(p2), smooth(false)
	{
		this->color = color;
	}
	Triangle(glm::vec3 p0, glm::vec3  p1, glm::vec3 p2, Material material) : p0(p0), p1(p1), p2(p2), smooth(false)
	{
		this->material = material;
	}

	void setNormals(glm::vec3 normal0, glm::vec3 normal1, glm::vec3 normal2) {
		n0 = normal0;
		n1 = normal1;
		n2 = normal2;
		smooth = true;
	}

	Hit intersect(Ray ray) {

		Hit hit;
		hit.hit = false;

		// Edge vectors 
		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		glm::vec3 DcrossE2 = glm::cross(ray.direction, e2);
		float det = glm::dot(e1, DcrossE2);

		if (det != 0)
		{
			float invDet = 1.0 / det;

			// Distance from p0 to ray origin
			glm::vec3 s = ray.origin - p0;

			float u = glm::dot(s, DcrossE2) * invDet;
			if (u < 0 || u > 1)
				return hit;

			glm::vec3 ScrossE1 = glm::cross(s, e1);
			float v = glm::dot(ray.direction, ScrossE1) * invDet;
			if (v < 0 || u + v > 1)
				return hit;

			float t = glm::dot(e2, ScrossE1) * invDet;

			if (t > 0)
			{
				hit.hit = true;
				hit.distance = t;
				hit.intersection = ray.origin + t * ray.direction;

				if (smooth)
				{
					float b0 = 1 - u - v;
					float b1 = u;
					float b2 = v;

					glm::vec3 interpolatedNormal = glm::normalize(b0 * n0 + b1 * n1 + b2 * n2);

					hit.normal = interpolatedNormal;
				}
				else
				{
					hit.normal = glm::normalize(glm::cross(e1, e2));
				}

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
		
		glm::vec3 d = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0); //implicit cast to vec3
		glm::vec3 o = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0); //implicit cast to vec3
		d = glm::normalize(d);
		
		
		float a = d.x*d.x + d.z*d.z - d.y*d.y;
		float b = 2 * (d.x * o.x + d.z * o.z - d.y * o.y);
		float c = o.x * o.x + o.z * o.z - o.y * o.y;
		
		float delta = b*b - 4 * a * c;
		
		if(delta < 0){
			return hit;
		}
		
		float t1 = (-b-sqrt(delta)) / (2*a);
		float t2 = (-b+sqrt(delta)) / (2*a);
		
		float t = t1;
		hit.intersection = o + t*d;
		if(t<0 || hit.intersection.y>1 || hit.intersection.y<0){
			t = t2;
			hit.intersection = o + t*d;
			if(t<0 || hit.intersection.y>1 || hit.intersection.y<0){
				return hit;
			}
		};
	
		hit.normal = glm::vec3(hit.intersection.x, -hit.intersection.y, hit.intersection.z);
		hit.normal = glm::normalize(hit.normal);
	
		
		Ray new_ray(o,d);
		Hit hit_plane = plane->intersect(new_ray);
		if(hit_plane.hit && hit_plane.distance < t && length(hit_plane.intersection - glm::vec3(0,1,0)) <= 1.0 ){
			hit.intersection = hit_plane.intersection;
			hit.normal = hit_plane.normal;
		}
		
		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0); //implicit cast to vec3
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0)); //implicit cast to vec3
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);
		
		return hit;
	}
};

class Mesh : public Object {
private:
	std::vector<Triangle> triangles;        ///< Triangles defined in object coordinates
	std::vector<glm::vec3> vertices;        ///< Vertex positions
	std::vector<glm::vec3> normals;         ///< Vertex normals
	bool smoothShading = true;              ///< Smooth shading enabled by default

public:
	// Load the mesh from an OBJ file with support for "s" smooth shading flag
	void loadOBJ(const std::string& filepath) {
		std::ifstream file(filepath);
		std::string line;

		while (std::getline(file, line)) {
			std::istringstream stream(line);
			std::string type;
			stream >> type;

			if (type == "v") {
				// Parse vertex position
				float x, y, z;
				stream >> x >> y >> z;
				vertices.push_back(glm::vec3(x, y, z));
			}
			else if (type == "vn") {
				// Parse vertex normal
				float nx, ny, nz;
				stream >> nx >> ny >> nz;
				normals.push_back(glm::vec3(nx, ny, nz));
			}
			else if (type == "f") {
				// Parse face indices; handle cases with or without normal indices
				std::vector<int> vertexIndices;
				std::vector<int> normalIndices;
				bool hasNormals = false;

				for (int i = 0; i < 3; ++i) {
					std::string vertexString;
					if (!(stream >> vertexString)) {
						std::cerr << "Error parsing face line: " << line << std::endl;
						return;
					}

					size_t pos = vertexString.find("//");
					if (pos != std::string::npos) {
						// Format: "v//n" (vertex with normal index but no texture coordinate)
						int vIndex = std::stoi(vertexString.substr(0, pos)) - 1;
						int nIndex = std::stoi(vertexString.substr(pos + 2)) - 1;
						vertexIndices.push_back(vIndex);
						normalIndices.push_back(nIndex);
						hasNormals = true;
					}
					else {
						// Format: "v" (only vertex index)
						int vIndex = std::stoi(vertexString) - 1;
						vertexIndices.push_back(vIndex);
					}
				}

				// Validate indices
				if (vertexIndices.size() != 3 || (hasNormals && normalIndices.size() != 3)) {
					std::cerr << "Face line has incorrect number of indices: " << line << std::endl;
					return;
				}
				for (int vIndex : vertexIndices) {
					if (vIndex < 0 || vIndex >= vertices.size()) {
						std::cerr << "Vertex index out of range: " << vIndex + 1 << " in line: " << line << std::endl;
						return;
					}
				}
				if (hasNormals) {
					for (int nIndex : normalIndices) {
						if (nIndex < 0 || nIndex >= normals.size()) {
							std::cerr << "Normal index out of range: " << nIndex + 1 << " in line: " << line << std::endl;
							return;
						}
					}
				}

				// Create triangle with vertex positions and handle normals based on shading mode
				Triangle triangle(vertices[vertexIndices[0]], vertices[vertexIndices[1]], vertices[vertexIndices[2]]);
				if (smoothShading && hasNormals) {
					// Use interpolated normals for smooth shading
					triangle.setNormals(normals[normalIndices[0]], normals[normalIndices[1]], normals[normalIndices[2]]);
				}
				else {
					// Compute flat normal if smooth shading is disabled or no normals provided
					glm::vec3 edge1 = vertices[vertexIndices[1]] - vertices[vertexIndices[0]];
					glm::vec3 edge2 = vertices[vertexIndices[2]] - vertices[vertexIndices[0]];
					glm::vec3 flatNormal = glm::normalize(glm::cross(edge1, edge2));
					triangle.setNormals(flatNormal, flatNormal, flatNormal);
				}
				triangles.push_back(triangle);
			}
			else if (type == "s") {
				// Smooth shading on/off
				std::string shading;
				stream >> shading;
				smoothShading = (shading != "off");
			}
		}
		file.close();
	}

	// Apply the transformation matrix to convert object coordinates to world coordinates
	Hit intersect(Ray ray) override {
		Hit closestHit;
		closestHit.hit = false;
		closestHit.distance = INFINITY;

		// Transform ray into object (local) coordinates using the inverse matrix
		glm::vec3 localOrigin = glm::vec3(inverseTransformationMatrix * glm::vec4(ray.origin, 1.0));
		glm::vec3 localDirection = glm::normalize(glm::vec3(inverseTransformationMatrix * glm::vec4(ray.direction, 0.0)));
		Ray localRay(localOrigin, localDirection);

		// Check intersections in object space
		for (Triangle& triangle : triangles) {
			Hit hit = triangle.intersect(localRay);
			if (hit.hit && hit.distance < closestHit.distance) {
				closestHit = hit;
			}
		}

		if (closestHit.hit) {
			// Convert intersection point and normal back to world coordinates
			closestHit.intersection = glm::vec3(transformationMatrix * glm::vec4(closestHit.intersection, 1.0));
			closestHit.normal = glm::normalize(glm::vec3(normalMatrix * glm::vec4(closestHit.normal, 0.0)));
			closestHit.distance = glm::distance(ray.origin, closestHit.intersection);
		}

		return closestHit;
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
//glm::vec3 ambient_light(0.1,0.1,0.1);
// new ambient light
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
		
        float r = glm::distance(point,lights[light_num]->position);
        r = max(r, 0.1f);
        color += lights[light_num]->color * (diffuse + specular) / r/r;
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

	
	Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.7f, 0.9f, 0.7f);
	green_diffuse.diffuse = glm::vec3(0.7f, 0.9f, 0.7f);

	Material red_specular;
	red_specular.ambient = glm::vec3(1.0f, 0.3f, 0.3f);
	red_specular.diffuse = glm::vec3(1.0f, 0.3f, 0.3f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.diffuse = glm::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;
	
	
	//Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = glm::vec3(0.3f, 1.0f, 0.3f);

	//Material red_specular;
	red_specular.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = glm::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	//Material blue_specular;
	blue_specular.ambient = glm::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;

	//objects.push_back(new Sphere(1.0, glm::vec3(1,-2,8), blue_specular));
	//objects.push_back(new Sphere(0.5, glm::vec3(-1,-2.5,6), red_specular));
	
	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));
	
    Material red_diffuse;
    red_diffuse.ambient = glm::vec3(0.09f, 0.06f, 0.06f);
    red_diffuse.diffuse = glm::vec3(0.9f, 0.6f, 0.6f);
        
    Material blue_diffuse;
    blue_diffuse.ambient = glm::vec3(0.06f, 0.06f, 0.09f);
    blue_diffuse.diffuse = glm::vec3(0.6f, 0.6f, 0.9f);
    objects.push_back(new Plane(glm::vec3(0,-3,0), glm::vec3(0.0,1,0)));
    objects.push_back(new Plane(glm::vec3(0,1,30), glm::vec3(0.0,0.0,-1.0), green_diffuse));
    objects.push_back(new Plane(glm::vec3(-15,1,0), glm::vec3(1.0,0.0,0.0), red_diffuse));
    objects.push_back(new Plane(glm::vec3(15,1,0), glm::vec3(-1.0,0.0,0.0), blue_diffuse));
    objects.push_back(new Plane(glm::vec3(0,27,0), glm::vec3(0.0,-1,0)));
    objects.push_back(new Plane(glm::vec3(0,1,-0.01), glm::vec3(0.0,0.0,1.0), green_diffuse));
	
	// Cones
	Material yellow_specular;
	yellow_specular.ambient = glm::vec3(0.1f, 0.10f, 0.0f);
	yellow_specular.diffuse = glm::vec3(0.4f, 0.4f, 0.0f);
	yellow_specular.specular = glm::vec3(1.0);
	yellow_specular.shininess = 100.0;
	
	/*Cone *cone = new Cone(yellow_specular);
	glm::mat4 translationMatrix = glm::translate(glm::vec3(5,9,14));
	glm::mat4 scalingMatrix = glm::scale(glm::vec3(3.0f, 12.0f, 3.0f));
	glm::mat4 rotationMatrix = glm::rotate(glm::radians(180.0f) , glm::vec3(1,0,0));
	cone->setTransformation(translationMatrix*scalingMatrix*rotationMatrix);
	objects.push_back(cone);
	
	Cone *cone2 = new Cone(green_diffuse);
	translationMatrix = glm::translate(glm::vec3(6,-3,7));
	scalingMatrix = glm::scale(glm::vec3(1.0f, 3.0f, 1.0f));
	rotationMatrix = glm::rotate(glm::atan(3.0f), glm::vec3(0,0,1));
	cone2->setTransformation(translationMatrix* rotationMatrix*scalingMatrix);
	objects.push_back(cone2);*/

	// Tringle
	//objects.push_back(new Triangle(glm::vec3(1, -3, 8), glm::vec3(3, -3, 6), glm::vec3(3, 0, 8), blue_specular);

	// Meshes
	Mesh* meshb = new Mesh();
	meshb->loadOBJ("./bunny.obj");

	//Set the object-to-world transformation
	glm::mat4 translationMatrixb = glm::translate(glm::vec3(-2.0f, -3.0f, 9.0f));
	glm::mat4 scalingMatrixb = glm::scale(glm::vec3(1.0));
	glm::mat4 rotationMatrixb = glm::rotate(glm::radians(0.0f), glm::vec3(0, 1, 0));

	//Combine transformations and set them in the mesh
	meshb->setTransformation(translationMatrixb * rotationMatrixb * scalingMatrixb);

	objects.push_back(meshb);

	Mesh* meshbn = new Mesh();
	meshbn->loadOBJ("./bunny_with_normals.obj");

	//Set the object-to-world transformation
	glm::mat4 translationMatrixbn = glm::translate(glm::vec3(2.0f, -3.0f, 9.0f));
	glm::mat4 scalingMatrixbn = glm::scale(glm::vec3(1.0));
	glm::mat4 rotationMatrixbn = glm::rotate(glm::radians(0.0f), glm::vec3(0, 1, 0));

	//Combine transformations and set them in the mesh
	meshbn->setTransformation(translationMatrixbn * rotationMatrixbn * scalingMatrixbn);

	objects.push_back(meshbn);
	
}
glm::vec3 toneMapping(glm::vec3 intensity){
	float gamma = 1.0/2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}
//int main(int argc, const char * argv[]) {
//
//    clock_t t = clock(); // variable for keeping the time of the rendering
//
//    int width = 1024; //width of the image
//    int height = 768; // height of the image
//    float fov = 90; // field of view
//
//	sceneDefinition(); // Let's define a scene
//
//	Image image(width,height); // Create an image where we will store the result
//	vector<glm::vec3> image_values(width*height);
//
//    float s = 2*tan(0.5*fov/180*M_PI)/width;
//    float X = -s * width / 2;
//    float Y = s * height / 2;
//
//    for(int i = 0; i < width ; i++)
//        for(int j = 0; j < height ; j++){
//
//			float dx = X + i*s + s/2;
//            float dy = Y - j*s - s/2;
//            float dz = 1;
//
//			glm::vec3 origin(0, 0, 0);
//            glm::vec3 direction(dx, dy, dz);
//            direction = glm::normalize(direction);
//
//            Ray ray(origin, direction);
//            image.setPixel(i, j, toneMapping(trace_ray(ray)));
//        }
//	
//    t = clock() - t;
//    cout<<"It took " << ((float)t)/CLOCKS_PER_SEC<< " seconds to render the image."<< endl;
//    cout<<"I could render at "<< (float)CLOCKS_PER_SEC/((float)t) << " frames per second."<<endl;
//
//	// Writing the final results of the rendering
//	if (argc == 2){
//		image.writeImage(argv[1]);
//	}else{
//		image.writeImage("./result.ppm");
//	}
//
//	
//    return 0;
//}

#include <thread>
#include <mutex>

const int NUM_THREADS = std::thread::hardware_concurrency();
mutex io_mutex;

void render_chank(int start, int end, int width, int height, float X, float Y, float s, vector<glm::vec3>& image_values) {
	for (int index = start; index < end; index++) {
		int i = index % width;
		int j = index / width;

		float dx = X + i * s + s / 2;
		float dy = Y - j * s - s / 2;
		float dz = 1;

		glm::vec3 origin(0, 0, 0);
		glm::vec3 direction(dx, dy, dz);
		direction = glm::normalize(direction);

		Ray ray(origin, direction);
		image_values[index] = toneMapping(trace_ray(ray));
	}
}

int main(int argc, const char* argv[]) {
	clock_t t = clock();

	int width = 1024;
	int height = 768;
	float fov = 90;

	sceneDefinition();

	Image image(width, height);
	vector<glm::vec3> image_values(width * height);

	float s = 2 * tan(0.5 * fov * M_PI / 180) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	// Determine segment size for each thread
	int num_pixels = width * height;
	int segment_size = num_pixels / NUM_THREADS;

	vector<thread> threads;
	for (int i = 0; i < NUM_THREADS; i++) {
		int start = i * segment_size;
		int end = (i == NUM_THREADS - 1) ? num_pixels : start + segment_size;
		threads.push_back(thread(render_chank, start, end, width, height, X, Y, s, ref(image_values)));
	}

	for (auto& thread : threads) {
		thread.join();
	}

	// Transfer image values to the image
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			image.setPixel(i, j, image_values[i + j * width]);
		}
	}

	t = clock() - t;
	cout << "Rendering time: " << ((float)t) / CLOCKS_PER_SEC << " seconds.\n";
	cout << "FPS: " << (float)CLOCKS_PER_SEC / ((float)t) << endl;

	if (argc == 2) {
		image.writeImage(argv[1]);
	}
	else {
		image.writeImage("./result.ppm");
	}

	return 0;
}