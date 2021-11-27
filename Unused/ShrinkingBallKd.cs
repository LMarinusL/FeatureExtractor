using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class ShrinkingBallKd : MonoBehaviour
{
    public KdTree<Transform> verticesKd = new KdTree<Transform>();
    public List<Transform> vertices2 = new List<Transform>();
    public List<Vector3> normals;
    public List<Vector3> vertices;
    public float initialRadius = 200.0f;
    public List<Vector3> verticesThin;
    public List<Vector3> MedialBallCenters;
    public List<float> MedialBallRadii;
    public GameObject dotred;
    public GameObject dotblue;
    public Mesh mesh;
    




    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.K))
        {
            getMesh();
        }
        if (Input.GetKey(KeyCode.I))
        {
            iterateVertices();
            InstantiatePoints();
        }
    }

    void getMesh()
    {
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        mesh = meshGenerator.mesh;
        vertices = mesh.vertices.ToList();
        normals = mesh.normals.ToList();
        for (int i = 0; i < mesh.vertices.Length -1; i++)
        {

            //var vert = new MeshComponent(i, mesh.normals[i], mesh.vertices[i]);
            GameObject star = Instantiate(dotblue, new Vector3(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z), new Quaternion(0, 0, 0, 0));
            Debug.Log(star);
            vertices2.Add(star.transform);
        }
        verticesKd.AddAll(vertices2);
    }

    bool checkRadius(int vertexIndex, float radius)
    {
        int iteration = 0;
        int maxIterations = 20;
        Vector3 medialBallCenter = vertices[vertexIndex] + normals[vertexIndex] * radius;
        while (iteration < maxIterations)
        {
            iteration++;
            Vector3 closest = verticesKd.FindClosest(medialBallCenter).position;
            if (Vector3.Distance(closest, medialBallCenter ) < radius && closest != vertices[vertexIndex])
            {
            return false;
            }
        }
        return true;
    }

    void getMedialBallCenter(int vertexIndex)
    {
        bool empty = false;
        float radius = initialRadius;
        while (empty == false)
        {
            radius -= 1.0f;
            empty = checkRadius(vertexIndex, radius);
            if (radius < 50f)
            {
                return;
            }
        }
        MedialBallCenters.Add(vertices[vertexIndex] + (normals[vertexIndex] * radius));
        Debug.Log(radius);
        MedialBallRadii.Add(radius);
    }

    void iterateVertices()
    {
        for (int i = 0; i < vertices.ToArray().Length; i++)
        {
            getMedialBallCenter(i);
        }
    }

    void InstantiatePoints()
    {
        for (int vertId = 0; vertId < MedialBallCenters.ToArray().Length; vertId++)
        {
            Instantiate(dotred, MedialBallCenters[vertId], transform.rotation);
            //Instantiate(dotred, new Vector3(MedialBallCenters[vertId].x, 300, MedialBallCenters[vertId].z), transform.rotation);
        }
    }
}
