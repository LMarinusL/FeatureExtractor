using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ToggleWater : MonoBehaviour
{
    [SerializeField] public GameObject water;
    // Start is called before the first frame update
    void Start()
    {

    }
    void Update()
    {
        if (Input.GetKey(KeyCode.B))
        {
            transform.position += new Vector3(0, 1, 0); ;
        }
        if (Input.GetKey(KeyCode.V))
        {
            transform.position += new Vector3(0, -1, 0); ;
        }


    }

}
