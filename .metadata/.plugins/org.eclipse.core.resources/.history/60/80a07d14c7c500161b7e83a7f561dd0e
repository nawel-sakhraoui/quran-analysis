'use strict';

var app = angular.module('QSIApp', ['angular-advanced-searchbox','ngMaterial']); 




app.controller('MainCtrl',  function($scope) {
	
		$scope.availableSearchParams = [
          { key: "السورة", name: "السورة", placeholder: "السورة" },
          { key: "مكان النزول", name: "مكان النزول", placeholder: "مكان النزول" },
          { key: "ترتيب السورة", name: "ترتيب السورة", placeholder: "ترتيب السورة" },
          { key: "الآية", name: "الآية", placeholder: "الآية", allowMultiple: true },
          
        ];
		
		$scope.data = {
		      group : '1'
	  	};

		  
		$scope.radioData = [
		      { label: 'بحث عام', value: "1" },
		      { label: '2', value: "2" },
		      { label: '3', value: "3", isDisabled: true },
		      { label: '4', value: "4" }
		 ];
		
		
		
		$scope.rest =""; 
	 	$scope.keyflag = 0;
	 	

	 	var elasticsearch = require('elasticsearch');
	 	var client = new elasticsearch.Client({
	 	  host: 'localhost:9200',
	 	  log: 'trace'
	 	});


	 	client.search({
	 	  index: 'quran-index',
	 	  type: 'surats',
	 	  body: {
	 	    query: {
	 	      match: {
	 	        body: '*'
	 	      }
	 	    }
	 	  }
	 	}).then(function (resp) {
	 	    $scope.hits = resp.hits.hits;
	 	}, function (err) {
	 	    console.trace(err.message);
	 	});
	 	
	 	
	 	
	 	
	 	$scope.checkIfEnterKeyWasPressed = function f1 ($event ){
		    var keyCode = $event.which || $event.keyCode;
		    if (keyCode === 13 ) {
		    	
		    	$scope.keyflag += 1;
				

		    	/*$http({
					method : "GET",
					url : 'views/about.html'//" http://localhost:9200/quran-index/_search?q=*:*",

		        	
		        	
				}).then(function mySucces(response) {
					$scope.rest  =    response.data;
				
				}, function myError(response) {
					$scope.rest  =  response.statusText;
				});*/
		    	
		    }
	 	};
	 	
	
	 	 
	 	  
	 	  
	 	$scope.$watch('data.group', function() {
	 		
	 		if (Object.keys($scope.searchParams).length !== 0 ) 
	 		
	 		{
	 			
	 		$scope.keyflag += 1;
	 			
			/*$http({
				method : "GET",
				url :"views/main.html"// "http://localhost:9200/quran-index/_search?q=*:*",

	        	
			}).then(function mySucces(response) {
				$scope.rest  =   response.data;
			
			}, function myError(response) {
				$scope.rest  =  response.statusText;
			});*/
	 		}
	 		
	 	});
	 	



});