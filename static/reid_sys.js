const titleEle = document.querySelector(".title-icon");
const sideBarEle = document.querySelector(".sidebar");
const input_imgEle = document.querySelectorAll(".picUpload");
const btnEle = document.getElementById("b1");
const selectEle0 = document.getElementById("s0");
const selectEle1 = document.getElementById("s1");
const selectEle2 = document.getElementById("s2");
const imgList1 = document.getElementById("show_img1");
const imgList2 = document.getElementById("show_img2");

titleEle.addEventListener("click", function () {
  sideBarEle.classList.toggle("active");
});

var model="";
var dataset = "";
var gallery_num = 0;

// 选择model
selectEle0.addEventListener("change", function (e) {
    model = e.target.value;
  // console.log(model);
});

// 选择dataset
selectEle1.addEventListener("change", function (e) {
    dataset = e.target.value;
  // console.log(dataset);
});

// 选择gallery_num
selectEle2.addEventListener("change", function (e) {
    gallery_num = e.target.value;
    console.log('gallery_num',gallery_num);
});


var formData = new FormData();
Array.from(document.querySelectorAll(".picUpload")).forEach((input_imgEle) => {
  input_imgEle.onchange = function () {
    let files = input_imgEle.files[0]; // 传入的图像文件
    const fileRead = new FileReader(); // 新建读取器
    fileRead.readAsDataURL(files); // 读取器读图像文件
    let that = this;
    if (this.id === 'i1') {
      formData.append("query_files", $("#i1")[0].files[0]);
    } else {
      formData.append("gallery_files", $("#i2")[0].files[0]);
    }
    console.log(formData)
    fileRead.onload = function () {
      let imgEle = document.createElement("img");
      imgEle.src = this.result;
      imgEle.setAttribute("width", "100px");
      // console.log(this, that);
      that.parentElement.parentElement.lastElementChild.append(imgEle);
    };
  };
});

function clearImg(ele){
  ele.innerHTML = '';
}

function addImg(data,queryDom,galleryDom) {

  if(data.query_list) {
    clearImg(queryDom);
    data.query_list.forEach(imgSrc=>{
      const img = document.createElement('img');
      img.src = `static/query/${imgSrc}`;
      queryDom.append(img);
    })
  }
  if (data.gallery2dis) {
    clearImg(galleryDom)
      for(let key in data.gallery2dis) {
        const img = document.createElement('img');
        img.src = `static/gallery/${key}`;
        const divEle = document.createElement('div');
        divEle.append(img);
        const spanEle = document.createElement('span');
        spanEle.innerText = data.gallery2dis[key];
        divEle.append(spanEle);
        divEle.classList.add('img-container');
        galleryDom.append(divEle);
      }
  }
}

// 上传图片到后端
btnEle.addEventListener("click", function (e) {
  e.preventDefault();
  // 从表单中获取文件对象
  // var image1 = $("#i1")[0].files[0];
  // var image2 = $("#i2")[0].files[0];
  // console.log("image1", image1);
  // console.log("image2", image2);
  // 创建 FormData 对象
  // console.log($("#i1")[0].files)
  // for (let i = 0; i < $("#i1")[0].files; i++) {
  //     formData.append("query_files", $("#i1")[0].files[i]);
  // }
  // for (let i = 0; i < $("#i2")[0].files; i++) {
  //     formData.append("gallery_files", $("#i2")[0].files[i]);
  // }
  if (!model || !dataset || !gallery_num) {
    alert('No model, dataset, or gallery_num selected');
    return;
  }
  formData.append("model", model);
  formData.append("dataset", dataset);
  formData.append("gallery_num", gallery_num);

  // 发起 Ajax 请求
  $.ajax({
    url: "/upload_file/",
    type: "POST", // 请求类型
    data: formData, // 发送的数据，即上传的文件数组
    dataType: "json", // 返回的数据类型
    processData: false, // 不需要将 formData 转换成字符串
    contentType: false, // 不需要设置请求头的 Content-Type
    success: function (res) {
      console.log(res);
      if(res.success) {
        console.log('success......')
        addImg(res.data,imgList1,imgList2);
        alert("上传成功！");
      } else {
        alert(res.message);
      }
    },
    error: function () {
      alert("上传失败！");
    },
  });
});
