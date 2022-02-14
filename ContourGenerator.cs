using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System.Linq;
using System;
using Unity.Mathematics;
public class ContourGenerator : MonoBehaviour
{

    public Grid grid;
    public int xSize;
    public int zSize;
    public GameObject dotone;
    public GameObject dottwo;
    public GameObject dotthree;
    public GameObject lineObject;
    public Vector3[] vectorList;
    public List<float3> contourVertices;
    public LineRenderer line;
    public Material material;
    public Vector3[] orderedArray;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.H))
        {
            getData();
            getContourVertices(100f, dotone);
            //getContourVertices(110f, dottwo);
            //getContourVertices(90f, dotthree);
            createLine(vectorList);
        }

    }

    void getData()
    {
        GameObject gridcreator = GameObject.Find("GridCreator");
        CreateGrid gridImporter = gridcreator.GetComponent<CreateGrid>();

        grid = gridImporter.grid;
        xSize = gridImporter.xSize;
        zSize = gridImporter.zSize;

        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
    }

    void getContourVertices(float height, GameObject dot)
    {
        Cell currentCell = grid.cells[0];
        Cell previousCell;
        Cell sideCell; 
        float xStep = grid.cells[4 + xSize + 1].x - grid.cells[4].x;
        float zStep = grid.cells[4 + xSize + 1].z - grid.cells[4].z;
        foreach (Cell cell in grid.cells)
        {
            previousCell = currentCell;
            currentCell = cell;

            if (currentCell.y == 0 || previousCell.y == 0 ||
                Mathf.Pow(Mathf.Pow(currentCell.x - previousCell.x, 2) + Mathf.Pow(currentCell.z - previousCell.z, 2), 0.5f) > 2 * xStep) { continue; }
            if ((currentCell.y <= height && previousCell.y > height) ||
                (currentCell.y >= height && previousCell.y < height))
            {
                float ratio = ((currentCell.y - height) / (previousCell.y - currentCell.y));
                contourVertices.Add(new float3(currentCell.x + (xStep * ratio), height,
                    currentCell.z));
            }
        }
          /*  foreach (Cell cell in grid.cells)
            {
                currentCell = cell;

                if (cell.index < grid.cells.Length - xSize -1)
            {
                sideCell = grid.cells[cell.index + xSize];
                if (currentCell.y == 0 || sideCell.y == 0 ||
                Mathf.Pow(Mathf.Pow(currentCell.x - sideCell.x, 2) + Mathf.Pow(currentCell.z - sideCell.z, 2), 0.5f) > 2 * zStep) { continue; }
                if ((currentCell.y <= height && sideCell.y > height) ||
                    (currentCell.y >= height && sideCell.y < height))
                {
                    float ratio = ((currentCell.y - height) / (sideCell.y - currentCell.y));
                    contourVertices.Add(new float3(currentCell.x, height,
                        currentCell.z + (zStep * ratio)));
                }
            }
        } */
        Debug.Log("contour: "+ contourVertices.ToArray().Length);
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        vectorList = meshGenerator.float3ToVector3Array(contourVertices.ToArray());
        //foreach (Vector3 point in vectorList)
        //{
        //    Instantiate(dot, point, transform.rotation);
        //}

    }

    void findTriangle(float contourHeight)
    {
        // find triangles crossing contour height 
    }

    void followHeight(Triangle start, float contourHeight)
    {

        // start at triangle with one vertex above/below and two at other side of height line
        // at the side of the triangle where there is one vertex below and one above,
            // add vertex and move to the adjacent triangle
        // nowagain check which side of triengle is on the border, add vertex, and move to adjacent triangle of that edge
    }

    void createLine(Vector3[] vectorList)
    {
        lineObject = new GameObject("Line");
        line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 2f;
        line.endWidth = 2f;
        line.positionCount = vectorList.Length;
        line.material = material;
        for (int i=0; i < vectorList.Length; i++)
        {
            line.SetPosition(i, vectorList[i]);
        }
    }
}
