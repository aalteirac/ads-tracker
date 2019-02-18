import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';
import {Tensor} from 'deeplearn';
import {IndexedDB} from './indexedDB.js';

const NB_CLASSIFS = 2;
const IMAGE_SIZE = 227;

const TOPK = 5;
var last = 0;

class Main{
  constructor(){
    this.infoTexts = [];
    this.inputTexts = [];
    this.buttons = [];
    this.training = -1;
    this.videoPlaying = true;
    this.tm;
    this.curIndex;
    this.knn = new KNNImageClassifier(NB_CLASSIFS, TOPK);

    this.inputTexts.push("NOPUB");
    this.inputTexts.push("PUB");

    this.displayChannels();
    this.knn.load()
        .then(() => {
          try {
            this.getVal("cached").then((cc)=>{
              if(cc===true){
                this.getModelFromDB();
                document.getElementById("predict").innerText= "Computing...";
                this.start()
              }
              else{  
                this.loadStaticModel().then(()=>{
                  this.start()
                  this.saveModelToDB();
                  document.getElementById("predict").innerText= "Computing...";
                })
              }
            })
          } catch (error) {
            alert("OPPPS");
          }
          
        });
  }
  httpGet(theUrl) {
    return new Promise((resolve, reject)=> {
      var xmlHttp = new XMLHttpRequest();
      xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
          resolve(xmlHttp.responseText);
      }
      xmlHttp.onerror = function () {
        reject("ko");
      };
      xmlHttp.open("GET", theUrl, true);
      xmlHttp.setRequestHeader('Access-Control-Allow-Origin', '*')
      xmlHttp.send(null);
    })
  }

  displayChannels(){
    this.getChannels("channels.lst").then((res)=>{
        res=res.split("#EXTINF");
        res.map((el)=>{
          var arr=el.split("\n");
          if(arr.length>2){
            var url=arr[1];
            var chid=url.split('/')[5].split("?")[0];
            var name=arr[0].split(",")[1];
            var pic=arr[0].split(" ")[1].split("\"")[1];  
            const div = document.createElement('div');
            var elem = document.createElement("img");
            elem.setAttribute("class","mych")
            elem.setAttribute("height", "100");
            elem.setAttribute("width", "130");
            elem.setAttribute("src", pic);
            elem.setAttribute("chid", encodeURI(chid));
            elem.setAttribute("tgt", encodeURI(url));
            elem.setAttribute("namech", encodeURI(name));
            div.appendChild(elem);
            div.addEventListener('click',(el)=>{
              this.videoPlaying=false;
              document.getElementById("predict") .innerText= "...Changing Channel...";
              var id=el.srcElement.getAttribute("chid");
              this.httpGet(`/chnl?id=${id}`).then(()=>{
                document.getElementById('lv').style.display="none";
                setTimeout(() => {
                  document.getElementById("predict").innerText= "Computing...";
                  document.getElementById('lv').setAttribute("src",'/lv?'+Math.random());
                  document.getElementById('lv').style.display="block";
                  this.videoPlaying=true;
                }, 5500);
                
              })
            })
            document.getElementById("ch").appendChild(div);
          }
        })    
    });
  }
  saveModelToDB(){
    this.save("cached",true);
    this.save("trainedSampleCount", this.knn.classExampleCount)
    this.save("trained", this.knn.classLogitsMatrices)
    for(let i=0;i<this.knn.classLogitsMatrices.length; i++){
      this.save("trainedValue"+i, this.knn.classLogitsMatrices[i]!=null?this.knn.classLogitsMatrices[i].dataSync():null)
      this.save("trainedValueLabel"+i, this.inputTexts[i])
    }
  }

  getModelFromDB(){
      this.getVal("trainedSampleCount").then((res)=>{
        var prom=[]
        for(let i=0;i<res.length; i++){
          if(res[i]>0)
            prom.push(this.getVal("trainedValue"+i));
        }
        Promise.all(prom).then((values)=>{
          this.getVal("trained").then((vl)=>{
            values.map((el,id)=>{
              this.knn.classLogitsMatrices[id]=new Tensor (vl[id].shape, vl[id].dtype,Object.values(el), vl[id].dataId);
              this.getVal("trainedValueLabel"+id).then((lab)=>{
                this.inputTexts[id]=lab;
              })

            })
            this.knn.classExampleCount=  res;
          })
        });
      })
  }

  loadStaticModel(){
    return new Promise((resolve,reject)=>{
      this.getFileModel("Classifier_1_values_data.json").then((val0)=>{
        this.getFileModel("Classifier_2_values_data.json").then((val1)=>{
          this.getFileModel("Classifier_Sampling_data.json").then((sp)=>{
            this.getFileModel("Classifiers_matrices_data.json").then((sv)=>{
              sv=JSON.parse(sv)
              sp=JSON.parse(sp)
              val0=JSON.parse(val0);
              val1=JSON.parse(val1);
              this.knn.classLogitsMatrices[0]=new Tensor (sv[0].shape, sv[0].dtype,Object.values(val0), sv[0].dataId);
              this.knn.classLogitsMatrices[1]=new Tensor (sv[1].shape, sv[1].dtype,Object.values(val1), sv[1].dataId);
              this.knn.classExampleCount=  sp;
              resolve();
              //alert("done");
            })
          })
        })
      })
    })
  }

  getChannels(fName){
    return new Promise((resolve, reject)=>{
      var req = new XMLHttpRequest();
      req.open("GET", fName, true);
      req.onreadystatechange = function() {
        if (req.readyState === 4) {
          if (req.status === 200) {
            var allText = req.responseText;
            resolve(allText)
          }
        }
      }
      req.send(null);
    })
  }

  getFileModel(fName){
    return new Promise((resolve, reject)=>{
      var req = new XMLHttpRequest();
      req.open("GET", "model/"+fName, true);
      req.onreadystatechange = function() {
        if (req.readyState === 4) {
          if (req.status === 200) {
            var allText = req.responseText;
            resolve(allText)
          }
        }
      }
      req.send(null);
    })
  }

  httpGet(theUrl) {
    return new Promise((resolve, reject)=> {
      var xmlHttp = new XMLHttpRequest();
      xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
          resolve(xmlHttp.responseText);
      }
      xmlHttp.onerror = function () {
        reject("ko");
      };
      xmlHttp.open("GET", theUrl, true);
      xmlHttp.setRequestHeader('Access-Control-Allow-Origin', '*')
      xmlHttp.send(null);
    })
  }

  createDLLink(obj,label){
    var blob = new Blob([ JSON.stringify(obj) ], {
      type : "text/plain;charset=utf-8;"
    });
    var modUrl = URL.createObjectURL(blob);

    //var data = "text/plain;charset=utf-8," + encodeURIComponent(JSON.stringify(obj));
    var a = document.createElement('a');
    a.href = modUrl;
    a.download = label+'_data.json';
    a.innerHTML = `download ${label}`;
    var container = document.getElementById('container');
    container.appendChild(a);
    container.appendChild(document.createElement('br'));
  }

  save(nm,val){
    var indexed=new IndexedDB();
    indexed.openDB().then((r)=>{
      indexed.store(nm,val);
    },(t)=>{
      console.log("KO",t);
    });
  }

  getVal(nm){
    var indexed=new IndexedDB();
    return indexed.openDB().then((r)=>{
      return indexed.getValue(nm);
    },(t)=>{
      console.log("KO",t);
    });
  }

  exportModel(){
    this.createDLLink(this.knn.classLogitsMatrices, "Classifiers_matrices");
    this.createDLLink(this.knn.classExampleCount, "Classifier_Sampling");
    this.createDLLink(this.knn.classLogitsMatrices[0].dataSync(), "Classifier_1_values");
    this.createDLLink(this.knn.classLogitsMatrices[1].dataSync(), "Classifier_2_values");
  }

  initStream(){
 
  } 
  initCam(ck){
    this.stop();
    var cm=ck?"user":"environment"
    const constraints = {
      advanced: [{
        facingMode: cm
      }]
    };
    navigator.mediaDevices.getUserMedia({video: constraints, audio: false})
        .then((stream) => {
          this.video.srcObject = stream;
          this.video.width = IMAGE_SIZE;
          this.video.height = IMAGE_SIZE;

          this.video.addEventListener('playing', ()=> this.videoPlaying = true);
          this.video.addEventListener('paused', ()=> this.videoPlaying = false);
          this.start();
        })

  }

  start(){
    if (this.timer) {
      this.stop();
    }
    try {
      this.videoPlaying = true
      //this.video.play();
    } catch (error) {
      console.log(error)
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
  
  checkStability(id,text,conf){
    //this.curIndex=id;
    if(this.curIndex!=id) {
      this.tm=new Date().valueOf();
      this.curIndex=id
    }
    else {
      var dd=new Date().valueOf()-this.tm;
      if(dd>3000){
        this.tm=new Date().valueOf();
        //console.log(text);
        if (text=="PUB"){
          document.getElementById("lay").style.background = "red";
        }
        else{
          document.getElementById("lay").style.background = "blue";
        }
        document.getElementById("predict") .innerText= text;
        //this.httpGet("http://localhost:8099/message?mess="+encodeURI(text))
      }
    }
    return 
  }

  animate(now){
    if(this.videoPlaying && (!last || now - last >= 2*1000)) {
      last = now;
      var image;
      var canvas = document.createElement('canvas');
      var context = canvas.getContext('2d');
      var img = document.getElementById('lv');
      canvas.width = img.width;
      canvas.height = img.height;
      context.drawImage(img, 0, 0 );
      var myData = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
      try {
        image = dl.fromPixels(myData);
      } catch (error) {
        console.log(error)
        this.timer = requestAnimationFrame(this.animate.bind(this));
        return;  
      }
      const exampleCount = this.knn.getClassExampleCount();
      if(Math.max(...exampleCount) > 0){
        this.knn.predictClass(image)
            .then((res)=>{
              for(let i=0;i<NB_CLASSIFS; i++){
                if(res.classIndex == i  && this.training==-1 && res.confidences[i]>0.95){
                  this.checkStability(res.classIndex,this.inputTexts[i],res.confidences[i].toFixed(2)*100);
                }
                // if(exampleCount[i] > 0){
                //   this.infoTexts[i].innerText = ` ${exampleCount[i]} samples - ${res.confidences[i].toFixed(2)*100}%`
                // }
              }
            })
            .then(()=> image.dispose())
      } else {
        image.dispose()
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

setTimeout(() => {
  new Main();
}, 2000);













