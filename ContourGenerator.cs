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
    public List<Face> faces;
    public List<Contour> contours;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.H))
        {
            getData();

            for (float j = 70f;j < 110f; j+=2f)
            {
                for (int i = 0; i < 40; i++)
                {
                    contourSegment(j);
                    contourSegment(j);
                }
            }
            for (int k = 0; k < contours.Count; k++)
            {
                createLine(contours[k]);
            }
        }
        /*if (Input.GetKeyDown(KeyCode.J))
        {
            connectCells();
        }*/
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
        contours = new List<Contour>();

    }

    int findFace(float height)
    {
        int toCheck = 0;
        bool found = false;
        int iter = 0;
        while (found == false && iter<10000)
        {

            toCheck = UnityEngine.Random.Range(0, grid.cells.Length);
            if(grid.cells[toCheck].attachedFaces[0].onContourLine != 0) 
            {
                iter++;
                continue; 
            }
            float height1 = grid.cells[toCheck].attachedFaces[0].startVertex.y;
            float height2 = grid.cells[toCheck].attachedFaces[0].endVertex.y;
            if ((height1 >= height && height2 < height) || (height1 < height && height2 >= height))
            {
                found = true;
            }
            iter++;
        }
        return toCheck;
    }

    void contourSegment(float height)
    {
        int vert = findFace(height);
        int count = 0;
        Contour contour;
        Contour contourTemp;
        List<Face> listFaces = new List<Face>();
        List<Face> outputList;
        int maxCount = 1000;
        outputList = followHeight(grid.cells[vert].attachedFaces[0], height, count, maxCount, listFaces);
        bool contourExists = false;
        contourTemp = new Contour(444, 444f, grid);
        foreach (Contour cont in contours)
        {
            if(height == cont.height)
            {
                contourExists = true;
                contourTemp = cont;
                break;
            }
        }
        if (!contourExists) {
            contour = new Contour(vert, height, grid);
        }
        else
       {
            contour = contourTemp;
      }
       if (contourExists)
       { 
           contour.addToList(outputList);
       }
       else
       {
            contour.faces = outputList;
       }
        int iteration = 0;
        int maxIterations = 20;
        while (contour.faces[0].ownTriangle.index != contour.faces[contour.faces.Count - 1].ownTriangle.index && iteration < maxIterations)
        {
            List<Face> listFaceAdd = new List<Face>();
            int newcount = 0;
            List<Face> outputListAdd;
            outputListAdd = followHeight(outputList[outputList.Count - 1], height, newcount, maxCount, listFaceAdd);
            contour.addToList(outputListAdd);
            iteration++;
        }
        contour.faceToVertex();
        if (!contourExists)
        {
            contours.Add(contour);
        }
    }

    public List<Face> followHeight(Face start, float contourHeight, int count, int maxCount, List<Face> facesOnHeight)
    {
        facesOnHeight.Add(start);
        count++;
        try
        {
            if (((start.index == facesOnHeight[0].index || start.faceTwin.index == facesOnHeight[0].index) && count > 5) || count > maxCount)
            {
                return facesOnHeight;
            }
            else
            {
                if ((start.next().endVertex.y >= contourHeight && start.next().startVertex.y < contourHeight) ||
                    (start.next().endVertex.y < contourHeight && start.next().startVertex.y >= contourHeight))
                {
                    return followHeight(start.next().faceTwin, contourHeight, count, maxCount, facesOnHeight);
                }
                else
                {
                    if ((start.previous().endVertex.y >= contourHeight && start.previous().startVertex.y < contourHeight) || (
                   start.previous().endVertex.y < contourHeight && start.previous().startVertex.y >= contourHeight))
                    {
                        return followHeight(start.previous().faceTwin, contourHeight, count, maxCount, facesOnHeight);
                    }
                    else
                    {
                        return facesOnHeight;
                    }
                }
            }
        }
        catch
        {
            //Debug.Log(" an error ocurred here");
            facesOnHeight.RemoveAt(facesOnHeight.Count - 1);
            return facesOnHeight;
        }
    }

    void createLine(Contour contour)
    {
        lineObject = new GameObject("Line");
        line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 0.5f;
        line.endWidth = 0.5f;
        line.positionCount = contour.cells.Count;
        line.material = material;
        line.material.color = new Color(0f, 0f, 0f, 1f);
        for (int i = 0; i < contour.cells.Count; i++)
        {
            line.SetPosition(i, contour.cells[i].position);
            contour.cells[i].computeCurvature(1);
        }
    }

    void connectCells()
    {
        foreach (Cell cell in grid.cells)
        {
            if(cell.y != 0)
            {
                Contour closestContour = null;
                float heightDiffMin = 100f;
                foreach(Contour contour in contours)
                {
                    if(Mathf.Pow(Mathf.Pow((contour.height - cell.y),2),0.5f) < heightDiffMin)
                    {
                        closestContour = contour;
                        heightDiffMin = contour.height - cell.y;
                    }
                }

                ContourCell closestCell = null;
                float cellDistMin = 1000f;
                foreach (ContourCell contourCell in closestContour.cellsArray)
                {
                    float currentDistance = Vector3.Distance(cell.position, contourCell.position);
                    if(currentDistance < cellDistMin)
                    {
                        cellDistMin = currentDistance;
                        closestCell = contourCell;
                    }
                }
                cell.contourCell = closestCell;
            }
        }
        Debug.Log("done");
    }
}

public class Contour : Component
{
    public int index;
    public float height;
    public List<Face> faces;
    public List<ContourCell> cells;
    public ContourCell[] cellsArray;
    Grid grid;

    public Contour(int i, float h, Grid gridLink)
    {
        index = i;
        height = h;
        grid = gridLink;
        faces = new List<Face>();
    }

    public void addToList(List<Face> toAdd)
    {
        faces.AddRange(toAdd);
    }

    public void faceToVertex()
    {
        cells = new List<ContourCell>();
        int i = 0;
        foreach (Face face in this.faces)
        {
            if (face.startVertex.y != 0 && face.endVertex.y != 0)
            {
                float xStep = face.startVertex.x - face.endVertex.x;
                float zStep = face.startVertex.z - face.endVertex.z;
                face.onContourLine = this.index;
                float ratio = ((face.startVertex.y - height) / (face.endVertex.y - face.startVertex.y));
                ContourCell newCell = new ContourCell(i * 10000 + face.index, new Vector3(face.startVertex.x + (xStep * ratio), height, face.startVertex.z + (zStep * ratio)), this, i);
                face.startVertex.contourCell = newCell;
                this.cells.Add(newCell);
                i++;
            }
        }
        cellsArray = cells.ToArray();
    }
}

public class ContourCell : Component
{
    int index;
    int contourIndex;
    public Vector3 position;
    Contour contour;
    int positionOnContour;
    public float curvature;

    public ContourCell(int i, Vector3 pos, Contour cont, int posOnCont)
    {
        index = i;
        position = pos;
        positionOnContour = posOnCont;
        contour = cont;
        contourIndex = cont.index;
    }

    public void computeCurvature(int spacing)
    {
        try
        {
            if (this.positionOnContour > 0 && this.positionOnContour < this.contour.cells.Count)
            {
                Vector3 expectedLocation = new Vector3(((this.contour.cells[this.positionOnContour - spacing].position.x - this.contour.cells[this.positionOnContour + spacing].position.x) /2)+ this.contour.cells[this.positionOnContour + spacing].position.x,
                                                        ((this.contour.cells[this.positionOnContour - spacing].position.y - this.contour.cells[this.positionOnContour + spacing].position.y)/2)+ this.contour.cells[this.positionOnContour + spacing].position.y,
                                                        ((this.contour.cells[this.positionOnContour - spacing].position.z - this.contour.cells[this.positionOnContour + spacing].position.z)/2)+ this.contour.cells[this.positionOnContour + spacing].position.z);
                this.curvature = Mathf.Pow(Mathf.Pow(Vector3.Distance(this.position, expectedLocation), 2), 0.5f);
            }
        }
        catch
        {
            Debug.Log(" error in curvatureComp");
        }
    }
}