using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveObject : MonoBehaviour
{

    // Start is called before the first frame update  
    void Start()
    {

    }

    // Update is called once per frame  
    void Update()
    {
        if (Input.GetKey(KeyCode.A))
        {
            transform.Translate(-0.5f, 0f, 0f);
        }
        if (Input.GetKey(KeyCode.D))
        {
            transform.Translate(0.5f, 0f, 0f);
        }
        if (Input.GetKey(KeyCode.S))
        {
            transform.Translate(0.0f, 0f, -0.5f);
        }
        if (Input.GetKey(KeyCode.W))
        {
            transform.Translate(0.0f, 0f, 0.5f);
        }
        if (Input.GetKey(KeyCode.X))
        {
            transform.Translate(0.0f, -0.5f, 0f);
        }
        if (Input.GetKey(KeyCode.Z))
        {
            transform.Translate(0.0f, 0.5f, 0f);
        }
    }
}